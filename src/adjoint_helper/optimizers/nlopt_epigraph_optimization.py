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

import nlopt  # type: ignore
import numpy as np
import numpy.typing as npt

from ..core.constraints import (
    line_width_and_spacing_constraint,
    connectivity_constraint,
    filter_and_project,
    tensor_jacobian_product,  # type: ignore
)
from ..core.objective_factory import get_physics_objective
from ..core.defs import ObjectiveReturn, PhysicsObjective
from ..core.export_settings import OptimizationSettings, SimulationSettings
from ..utils.util import save_output


class NloptEpigraphOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `nlopt` backend for non-differentiable/multi-objective optimizations. Uses
    the epigraph formulation to recast the problem in a differentiable manner.
    (see: https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/#equivalent-formulations-of-optimization-problems)
    """

    linewidth_tol: float
    connectivity_tol: float
    optimizer: nlopt.opt
    n_objectives: int
    epigraph_min: float
    epigraph_max: float

    def __init__(
        self,
        optimizer: nlopt.opt,  # Many of these algorithms depend on system size, so you must supply your own
        n_objectives: int,
        epigraph_min: float = 0,
        epigraph_max: float = 1,
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

        if n_objectives <= 0:
            raise ValueError("Epigraph formulation requires at least one objective")

        self.n_objectives = n_objectives

        self.epigraph_min = epigraph_min
        self.epigraph_max = epigraph_max

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
        weights: npt.NDArray[np.float64],
        grad: npt.NDArray[np.float64],
        settings: SimulationSettings,
    ) -> float:
        spacing, grad[:] = line_width_and_spacing_constraint(  # type: ignore
            weights=weights, settings=settings, optimization=self
        )

        return spacing  # type: ignore

    def _connectivity_cons(
        self,
        weights: npt.NDArray[np.float64],
        grad: npt.NDArray[np.float64],
        settings: SimulationSettings,
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
        connectivity, grad[:] = connectivity_constraint(weights, settings, self)

        return connectivity

    def _nlopt_epigraph_constraint(
        self,
        result: npt.NDArray[np.float64],
        epigraph_and_weights: npt.NDArray[np.float64],
        gradient: npt.NDArray[np.float64],
        settings: SimulationSettings,
    ) -> float:

        epigraph = epigraph_and_weights[0]
        weights = epigraph_and_weights[1:]

        opt = settings.get_objective(self)

        f0, grad = opt(weights)  # type: ignore

        if gradient.size > 0:
            gradient[:, 0] = -1  # gradient with respect to epigraph variable
            gradient[:, 1:] = grad[0][0].T  # real gradients per-parameter

        result[:] = np.real(f0) - epigraph

        self.obj.append(np.real(f0))  # type: ignore
        self.weights.append(epigraph_and_weights[1:].copy())
        self.data.append(epigraph)  # type: ignore

        print(f"Iteration: {len(self.weights)}, objective: {epigraph:.4e}\n")

        return f0[0]  # type: ignore

    def optimize(self, settings: SimulationSettings) -> npt.NDArray[np.float64]:
        """_summary_

        Args:
            settings (SimulationSettings): Simulation settings for this simulation
            optimization (OptimizationSettings): Optimization settings

        Returns:
            npt.NDArray[np.float64]: Optimal weights after the optimization
        """

        lb = np.zeros((settings.total_n()[0],))
        ub = np.ones((settings.total_n()[0],))
        weights = np.ones(settings.total_n()[0]) * 0.5  # add epigraph weight

        for mask in settings.border_masks(self.filter_radius):
            weights[mask.locations.flatten()] = mask.value

        history_fpath = settings.data_dir / settings.history_fname

        self.use_damping = True

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            weights = self.weights[-1]

        weights_and_epigraph = np.insert(
            weights, 0, (self.epigraph_max - self.epigraph_min) / 2
        )  # initial guess

        lb = np.insert(lb, 0, self.epigraph_min)
        ub = np.insert(ub, 0, self.epigraph_max)

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

            def mconstraint(
                result: npt.NDArray[np.float64],
                epigraph_and_weights: npt.NDArray[np.float64],
                gradient: npt.NDArray[np.float64],
            ):
                self._nlopt_epigraph_constraint(
                    result=result,
                    epigraph_and_weights=epigraph_and_weights,
                    gradient=gradient,
                    settings=settings,
                )

            solver.add_inequality_mconstraint(  # type: ignore
                mconstraint, np.array([1e-6] * self.n_objectives)
            )

            weights_and_epigraph[:] = solver.optimize(weights_and_epigraph)  # type: ignore

            save_output(
                weights_and_epigraph, settings, self, sigmoid_bias, history_fpath
            )
            self.last_completed_index = i

        optimal_design_weights = filter_and_project(
            weights[1:],
            settings,
            self,
        )

        save_output(
            weights_and_epigraph[1:], settings, self, 0, history_fpath, binarize=True
        )

        return optimal_design_weights


@get_physics_objective.register
def _(
    settings: SimulationSettings, optimization: NloptEpigraphOptimizationSettings
) -> PhysicsObjective:
    def get_nlopt_epigraph_objective(
        weights: list[npt.NDArray[np.float64]],
    ) -> ObjectiveReturn:
        """Default objective function used for `nlopt` optimizations. For simple
        (single-objective) optimizations, this is sufficient as-is. For
        more complicated objectives, you will need to customize it for your
        needs. Weights should not be updated in-place in this function.

        Args:
            weights (list[npt.NDArray[np.float64]]): List of weights per design region.

        Returns:
            out (ObjectiveReturn): FOM and gradients for the given weights
        """

        opt = settings.create_opt(optimization)

        obj_val, dJ_du = opt([filter_and_project(weights, settings, optimization)])  # type: ignore

        grad = np.zeros((settings.total_n()[0], optimization.n_objectives))
        for k in range(optimization.n_objectives):
            grad[:, k] = tensor_jacobian_product(filter_and_project, 0)(
                weights,
                settings,
                optimization,
                dJ_du[0][0][:, k],
            )

        return obj_val, [[grad]]

    return get_nlopt_epigraph_objective
