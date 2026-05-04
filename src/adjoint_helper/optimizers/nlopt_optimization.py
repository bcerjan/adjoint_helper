"""
Meep Adjoint Helper
Copyright (C) 2026 Ben Cerjan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from ..core.defs import PhysicsObjective, ObjectiveReturn, WeightsType, RawWeightsType
from ..core.export_settings import OptimizationSettings
from ..core.base_settings import SimulationSettingsBase
from ..core.constraints import tensor_jacobian_product  # type: ignore
from ..core.objective_factory import get_physics_objective
from ..utils.util import save_output, apply_masks

import numpy as np
import nlopt  # type: ignore


class NloptOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `nlopt` backend.

    Generally only good for single-objective optimizations.
    If you can convert your multi-objective FOM into a single one, this might work
    for you, otherwise consider using `NloptEpigraphOptimizationSettings` which
    handles multi-objectives that are not inherently differentiable better.
    """

    linewidth_tol: float
    connectivity_tol: float
    optimizer: nlopt.opt

    def __init__(
        self,
        optimizer: nlopt.opt,  # Many of these algorithms depend on system size, so you must supply your own
        minimum_size: float = 0.05,
        sigmoid_bias_threshold: float = 32,  # Sigmoid bias at which eps_avg turns on
        sigmoid_threshold: float = 0.5,  # Eta
        sigmoid_erosion: float = 0.65,  # Eta_e
        sigmoid_biases: list[float] = [4.0, 8, 16, 24, 32, 40],
        connectivity_sigmoid_threshold: float = 16,
        linewidth_sigmoid_threshold: float = 24,  # Sigmoid bias at which line width constraint turns on
        max_evals: list[int] | int = 10,  # if int, all biases get same number
        maximum_runtime: float = 200,
        minimum_runtime: float = 0,
        decay_by: float = 1e-6,
        use_smoothed_projection: bool = False,
        do_connectivity: bool = False,
        linewidth_tol: float = 1e-3,
        connectivity_tol: float = 1e-3,
    ):
        self.linewidth_tol = linewidth_tol
        self.connectivity_tol = connectivity_tol
        self.optimizer = optimizer

        super().__init__(
            minimum_size=minimum_size,
            sigmoid_bias_threshold=sigmoid_bias_threshold,
            sigmoid_threshold=sigmoid_threshold,
            sigmoid_erosion=sigmoid_erosion,
            sigmoid_biases=sigmoid_biases,
            connectivity_sigmoid_threshold=connectivity_sigmoid_threshold,
            linewidth_sigmoid_threshold=linewidth_sigmoid_threshold,
            max_evals=max_evals,
            maximum_runtime=maximum_runtime,
            minimum_runtime=minimum_runtime,
            decay_by=decay_by,
            use_smoothed_projection=use_smoothed_projection,
            do_connectivity=do_connectivity,
        )

    def _linewidth_cons(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettingsBase,
    ) -> float:

        spacing, grad[:] = settings.line_width_and_spacing(
            weights=weights, optimization=self
        )

        return spacing

    def _connectivity_cons(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettingsBase,
    ) -> float:
        """_summary_

        Args:
            weights (npt.NDArray[np.float64]): _description_
            grad (npt.NDArray[np.float64]): _description_
            settings (SimulationSettings): _description_
            optimization (OptimizationSettings): _description_

        Returns:
            float: _description_
        """
        connectivity, grad[:] = settings.connectivity_constraint(
            weights=weights, optimization=self
        )

        return connectivity

    def _nlopt_objective_f(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettingsBase,
    ) -> float:
        """
        Internal objective function, just calls the user-supplied objective function
        and returns a value from it. Takes the norm of the real parts of the
        objective function value as uses that as the return.

        Needs to be a minimization objective

        Updates gradient in-place
        """

        opt = settings.get_objective(self)

        f0, grad[:] = opt(weights)

        obj: float = np.linalg.norm(np.real(f0))  # type: ignore
        self.obj.append(obj)
        self.weights.append(weights.copy())
        self.data.append(obj)

        print(f"Iteration: {len(self.weights)}, objective: {obj:.4e}\n")
        print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}\n")

        return obj

    def optimize(self, settings: SimulationSettingsBase) -> WeightsType:
        """_summary_

        Args:
            settings (SimulationSettings): Simulation settings for this simulation
            optimization (OptimizationSettings): Optimization settings

        Returns:
            npt.NDArray[np.float64]: Optimal weights after the optimization
        """

        lb = np.zeros(settings.total_n_raw())
        ub = np.ones(settings.total_n_raw())
        weights = np.ones(settings.total_n_raw()) * 0.5

        apply_masks(masks=settings.get_masks(self.filter_radius), weights=weights)

        history_fpath = settings.data_dir / settings.history_fname

        self.use_damping = True

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            weights = self.weights[-1]

        for i, sigmoid_bias in enumerate(
            biases[last_index + 1 :], start=last_index + 1
        ):
            self.sigmoid_bias = sigmoid_bias
            solver = self.optimizer
            solver.set_lower_bounds(lb)  # type: ignore
            solver.set_upper_bounds(ub)  # type: ignore

            solver.set_maxeval(self.max_evals[i])  # type: ignore

            if self.apply_linewidth:
                solver.add_inequality_constraint(  # type: ignore
                    lambda x, g: self._linewidth_cons(x, g, settings),  # type: ignore
                    self.linewidth_tol,
                )

            if self.apply_connectivity:
                solver.add_inequality_constraint(  # type: ignore
                    lambda x, g: self._connectivity_cons(x, g, settings),  # type: ignore
                    self.connectivity_tol,
                )

            solver.set_param("dual_ftol_rel", 1e-8)  # type: ignore

            # Note: this needs to be set _after_ you set use_epsavg above, or that
            # doesn't work, and then things get bad.
            solver.set_min_objective(  # type: ignore
                lambda x, g: self._nlopt_objective_f(x, g)  # type: ignore
            )

            weights[:] = solver.optimize(weights)  # type: ignore

            save_output(weights, settings, self, sigmoid_bias, history_fpath)
            self.last_completed_index = i

        optimal_design_weights = settings.filter_and_project(
            weights=weights, optimization=self
        )

        save_output(weights, settings, self, 0, history_fpath, binarize=True)

        return settings.raw_to_weightslike(optimal_design_weights)


@get_physics_objective.register
def _(
    settings: SimulationSettingsBase, optimization: NloptOptimizationSettings
) -> PhysicsObjective:
    def get_nlopt_objective(
        weights: RawWeightsType,
    ) -> ObjectiveReturn:
        """Default objective function used for `nlopt` optimizations. For simple
        (single-objective) optimizations, this is sufficient as-is. For
        more complicated objectives, you will need to customize it for your
        needs. Weights should not be updated in-place in this function.

        Args:
            weights (RawWeightsType): List of weights concatenated together.

        Returns:
            out (ObjectiveReturn): FOM and gradients for the given weights
        """

        opt = settings.create_opt(optimization)

        obj_val, grad = opt(
            settings.filter_and_project(weights=weights, optimization=optimization)
        )

        grad[:] = tensor_jacobian_product(settings.filter_and_project, 0)(
            weights,
            optimization,
            grad,
        )

        return obj_val, grad

    return get_nlopt_objective
