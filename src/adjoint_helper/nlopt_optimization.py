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

from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.optimization_settings import OptimizationSettings
import numpy as np
import numpy.typing as npt
from adjoint_helper.constraints import (
    line_width_and_spacing_constraint,
    connectivity_constraint,
    filter_and_project,
)
from adjoint_helper.util import save_output
import nlopt  # type: ignore


class NloptOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `nlopt` backend.
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
        sigmoid_biases: list[float] = [4, 8, 16, 24, 32, 40],
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
        weights: npt.NDArray[np.float64],
        grad: npt.NDArray[np.float64],
        settings: SimulationSettings,
    ) -> float:
        spacing, grad[:] = line_width_and_spacing_constraint(  # type: ignore
            weights=weights, gradient=grad, settings=settings, optimization=self
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

    def _nlopt_objective_f(
        self,
        weights: npt.NDArray[np.float64],
        grad: npt.NDArray[np.float64],
        settings: SimulationSettings,
    ) -> float:
        """
        Default optimization function for nlopt calling.
        Can be customized if your `mpa` objective function has more returns.
        For simple cases (single freq. single `mpa` objective), this should be sufficient
        Needs to be a minimization objective
        """
        from adjoint_helper.constraints import filter_and_project
        from autograd import tensor_jacobian_product  # type: ignore

        opt = settings.create_opt(self)

        f0, dJ_du = opt([filter_and_project(weights, self, self)])  # type: ignore

        grad[:] = tensor_jacobian_product(filter_and_project, 0)(
            weights,
            self,
            self,
            dJ_du,
        )

        self.obj.append(np.real(f0))  # type: ignore
        self.weights.append(weights.copy())
        self.data.append(f0)  # type: ignore

        print(f"Iteration: {len(self.weights)}, objective: {f0[0]:.4e}\n")
        print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}\n")

        return f0[0]  # type: ignore

    def optimize(self, settings: SimulationSettings) -> npt.NDArray[np.float64]:
        """_summary_

        Args:
            settings (SimulationSettings): Simulation settings for this simulation
            optimization (OptimizationSettings): Optimization settings

        Returns:
            npt.NDArray[np.float64]: Optimal weights after the optimization
        """

        lb = np.zeros((settings.total_n(),))
        ub = np.ones((settings.total_n(),))
        weights = np.ones(settings.total_n()) * 0.5

        for mask in settings.border_masks(self):
            weights[mask.locations.flatten()] = mask.value

        history_fpath = settings.data_dir + settings.history_fname

        self.use_damping = True

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            weights = self.weights[-1]

        for i, sigmoid_bias in enumerate(
            biases[last_index + 1 :], start=last_index + 1
        ):
            self.apply_settings(sigmoid_bias)
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

        optimal_design_weights = filter_and_project(
            weights[:],
            settings,
            self,
        )

        save_output(weights, settings, self, 0, history_fpath, binarize=True)

        return optimal_design_weights
