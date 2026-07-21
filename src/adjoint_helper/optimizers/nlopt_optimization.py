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

from ..core.defs import (
    ObjectiveFunction,
    ObjectiveReturn,
    WeightsType,
    RawWeightsType,
    combine_gradients,
)
from ..core.base_settings import (
    SimulationSettings,
    OptimizationSettings,
    register_objective,
)
from ..utils.util import save_output

import numpy as np
import nlopt  # type: ignore
from pydantic import Field
from typing import List


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
        grad: RawWeightsType,
        settings: SimulationSettings,
    ) -> float:

        spacing, grad[:] = settings.line_width_and_spacing(optimization=self)

        return spacing

    def _connectivity_cons(
        self,
        grad: RawWeightsType,
        settings: SimulationSettings,
    ) -> float:

        connectivity, grad[:] = settings.connectivity_constraint(optimization=self)

        return connectivity

    def _nlopt_objective_f(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettings,
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
        self.weights.append(settings.vector_to_arrays(weights))
        self.data.append(obj)

        print(f"\nstep = {len(self.weights) + 1}")
        print(f"\tobjective = {np.linalg.norm(np.real(f0)):.4e}")
        print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}\n")
        if len(self.obj) > 1:
            diff = np.abs(self.obj[-1] - self.obj[-2])
            print(f"\tobjective diff: {diff}")
            if diff <= 1e-8:
                print(
                    "Objective did not change, something is wrong so we are aborting."
                )
                exit()

        return obj

    def optimize(self, settings: SimulationSettings) -> List[WeightsType]:
        """_summary_

        Args:
            settings (SimulationSettings): Simulation settings for this simulation
            optimization (OptimizationSettings): Optimization settings

        Returns:
            npt.NDArray[np.float64]: Optimal weights after the optimization
        """

        settings.calculate_normalization()

        lb = np.zeros(settings.total_n())
        ub = np.ones(settings.total_n())
        weights: RawWeightsType = np.ones(settings.total_n()) * 0.5

        settings.apply_masks()

        self.use_damping = True

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            last_weights = self.weights[-1]
            weights = settings.arrays_to_vector(last_weights)

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
                    lambda _, g: self._linewidth_cons(grad=g, settings=settings),  # pyright: ignore[reportUnknownArgumentType]
                    self.linewidth_tol,
                )

            if self.apply_connectivity:
                solver.add_inequality_constraint(  # type: ignore
                    lambda _, g: self._connectivity_cons(grad=g, settings=settings),  # pyright: ignore[reportUnknownArgumentType]
                    self.connectivity_tol,
                )

            solver.set_param("dual_ftol_rel", 1e-8)  # type: ignore

            # Note: this needs to be set _after_ you set use_epsavg above, or that
            # doesn't work, and then things get bad.
            solver.set_min_objective(  # type: ignore
                lambda x, g: self._nlopt_objective_f(x, g, settings)  # pyright: ignore[reportUnknownArgumentType]
            )

            weights[:] = solver.optimize(weights)

            save_output(settings, self, sigmoid_bias)
            self.last_completed_index = i

        optimal_design_weights = settings.filter_and_project(optimization=self)

        save_output(settings, self, 0, binarize=True)

        return optimal_design_weights


@register_objective(SimulationSettings, NloptOptimizationSettings)
def _(
    settings: SimulationSettings, optimization: NloptOptimizationSettings
) -> ObjectiveFunction:
    def get_nlopt_objective(
        weights: RawWeightsType,
    ) -> ObjectiveReturn:
        """Default objective function used for `nlopt` optimizations. For simple
        (single-objective) optimizations, this is sufficient as-is. For
        more complicated objectives, you might need to customize it for your
        needs. Weights should not be updated in-place in this function.

        Args:
            weights (List[WeightsType]): List of weights for each design region.

        Returns:
            out (ObjectiveReturn): FOM and gradients for the given weights
        """

        settings.update_weights(settings.vector_to_arrays(weights))
        settings.apply_masks()
        settings.apply_symmetry()

        opt = settings.create_opt(optimization)

        obj_val, grads = opt(settings.filter_and_project(optimization=optimization))

        grad_out = np.zeros((settings.total_n(),), dtype=np.float64)
        temp_grads = settings.get_empty_grads()

        for dr_i, r in enumerate(settings.design_regions):
            for obj_func in range(settings.n_objective_functions):
                for param in range(settings.n_design_parameters):
                    r.backpropagate(
                        grads[dr_i][obj_func][param], optimization=optimization
                    )
                    temp_grads[dr_i][obj_func][param][:] = grads[dr_i][obj_func][param]

        grad_out = combine_gradients(temp_grads)
        vec_out = np.concatenate(grad_out)

        return obj_val, vec_out

    return get_nlopt_objective
