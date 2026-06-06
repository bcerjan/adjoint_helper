"""
Adjoint Helper
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

__all__ = ["NloptOptimizationSettings"]

from ..core.defs import PhysicsObjective, ObjectiveReturn, WeightsType, RawWeightsType
from ..core.export_settings import OptimizationSettings
from ..core.base_settings import SimulationSettingsBase, register_physics_objective
from ..core.constraints import tensor_jacobian_product  # type: ignore
from ..utils.util import save_output, apply_masks, wrapped_filter_and_project

import numpy as np
import nlopt  # type: ignore
from pydantic import Field


class NloptOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `nlopt` backend.

    Generally only good for single-objective optimizations.
    If you can convert your multi-objective FOM into a single one, this might work
    for you, otherwise consider using `NloptEpigraphOptimizationSettings` which
    handles multi-objectives that are not inherently differentiable better.
    """

    linewidth_tol: float = 1e-3
    connectivity_tol: float = 1e-3
    optimizer: nlopt.opt = Field(exclude=True)
    sigmoid_biases: list[float] = [4, 8, 16, 32, 64, 128]

    def _linewidth_cons(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettingsBase,
    ) -> float:

        spacing, grad[:] = settings.line_width_and_spacing(
            weights=settings.raw_to_weightslike(weights), optimization=self
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
            weights=settings.raw_to_weightslike(weights), optimization=self
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

        print(f"\nstep = {len(self.weights) + 1}")
        print(f"\tobjective = {np.linalg.norm(np.real(f0)):.4e}")
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

        settings.calculate_normalization()

        lb = np.zeros(settings.total_n_raw())
        ub = np.ones(settings.total_n_raw())
        weights = np.ones(settings.total_n_raw()) * 0.5

        apply_masks(
            masks=settings.get_masks(self.filter_radius),
            weights=weights,
            multi_region=settings.is_multi_region(),
        )

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
                    lambda x, g: self._linewidth_cons(x, g, settings),  # pyright: ignore[reportUnknownArgumentType]
                    self.linewidth_tol,
                )

            if self.apply_connectivity:
                solver.add_inequality_constraint(  # type: ignore
                    lambda x, g: self._connectivity_cons(x, g, settings),  # pyright: ignore[reportUnknownArgumentType]
                    self.connectivity_tol,
                )

            solver.set_param("dual_ftol_rel", 1e-8)  # type: ignore

            # Note: this needs to be set _after_ you set use_epsavg above, or that
            # doesn't work, and then things get bad.
            solver.set_min_objective(  # type: ignore
                lambda x, g: self._nlopt_objective_f(x, g, settings)  # pyright: ignore[reportUnknownArgumentType]
            )

            weights[:] = solver.optimize(weights)  # type: ignore

            save_output(weights, settings, self, sigmoid_bias)
            self.last_completed_index = i

        optimal_design_weights = settings.filter_and_project(
            weights=weights, optimization=self
        )

        save_output(weights, settings, self, 0, binarize=True)

        return settings.raw_to_weightslike(optimal_design_weights)


@register_physics_objective(SimulationSettingsBase, NloptOptimizationSettings)
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
            settings.filter_and_project(
                weights=settings.raw_to_weightslike(weights), optimization=optimization
            )
        )

        # Squeeze grad into a 1D vector:
        grad = grad.ravel()
        print(f"grad: {grad.shape}, weights: {weights.shape}")
        grad[:] = tensor_jacobian_product(wrapped_filter_and_project, 0)(
            weights,
            settings,
            optimization,
            grad,
        )

        return obj_val, grad

    return get_nlopt_objective
