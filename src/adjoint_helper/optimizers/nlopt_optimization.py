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

from adjoint_helper.core.pydantic_defs import SolverConfig
from ..core.defs import (
    WeightsType,
    RawWeightsType,
)
from ..core.base_settings import (
    SimulationSettings,
    OptimizationSettings,
)
from ..utils.util import save_output

import numpy as np
import nlopt  # type: ignore
from pydantic import Field, computed_field
from typing import List
import logging

logger = logging.getLogger(__name__)


# TODO: Add dict of {param, val} for nlopt settings so people can set
# them manually?
class NloptOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `nlopt` backend.

    Generally only good for single-objective optimizations.
    If you can convert your multi-objective FOM into a single one, this might work
    for you, otherwise consider using `NloptEpigraphOptimizationSettings` which
    handles multi-objectives that are not inherently differentiable better.
    """

    linewidth_tol: float = 1e-3  # Value at which we consider the tolerance satisfied
    linewidth_min_grad: float = 1e-6  # value at which we treat the gradient as 0
    connectivity_tol: float = 1e-3
    connectivity_min_grad: float = 1e-6  # value at which we tread the gradient as 0
    optimizer: nlopt.opt = Field(exclude=True, repr=False)
    sigmoid_biases: list[float] = [4, 8, 16, 32, 64, 128]
    nlopt_params: dict = {}  # Can pass arbitrary settings through this to apply to the solver

    @computed_field
    @property
    def solver_config(self) -> SolverConfig:
        params_dict = {}
        for i in range(self.optimizer.num_params()):
            name = self.optimizer.nth_param(i)
            params_dict[name] = self.optimizer.get_param(name, 0.0)

        return SolverConfig(
            library="nlopt",
            algorithm=self.optimizer.get_algorithm_name(),
            params=params_dict,
        )

    def _linewidth_cons(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettings,
    ) -> float:

        settings.update_weights(settings.vector_to_arrays(weights))
        settings.apply_masks()
        settings.apply_symmetry()

        spacing, grad_out = settings.line_width_and_spacing(optimization=self)

        if grad_out.size > 0:
            grad[:] = grad_out

        return spacing

    def _connectivity_cons(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettings,
    ) -> float:

        settings.update_weights(settings.vector_to_arrays(weights))
        settings.apply_masks()
        settings.apply_symmetry()

        connectivity, grad_out = settings.connectivity_constraint(optimization=self)
        if grad_out.size > 0:
            grad[:] = grad_out

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
        f0, vec_out = opt(weights)

        if grad is not None and grad.size > 0:
            grad[:] = vec_out

        obj: float = np.sum(np.real(f0))
        self.obj.append(obj)
        self.weights.append(settings.vector_to_arrays(weights))
        self.data.append(obj)
        save_output(settings, self, self.sigmoid_bias)

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
        weights: RawWeightsType = np.concatenate(settings.weights_raw)

        settings.apply_masks()

        # Handle restarting:
        _, last_index = self._restart_bias_and_index()
        biases = self.sigmoid_biases

        for i, sigmoid_bias in enumerate(biases[last_index:], start=last_index):
            self.sigmoid_bias = sigmoid_bias
            solver = nlopt.opt(self.optimizer)
            solver.set_lower_bounds(lb)  # type: ignore
            solver.set_upper_bounds(ub)  # type: ignore

            solver.set_maxeval(self.max_evals[i])  # type: ignore

            if self.apply_linewidth:
                solver.add_inequality_constraint(  # type: ignore
                    lambda w, g: self._linewidth_cons(
                        weights=w, grad=g, settings=settings
                    ),  # pyright: ignore[reportUnknownArgumentType]
                    self.linewidth_tol,
                )

            if self.apply_connectivity:
                solver.add_inequality_constraint(  # type: ignore
                    lambda w, g: self._connectivity_cons(
                        weights=w, grad=g, settings=settings
                    ),  # pyright: ignore[reportUnknownArgumentType]
                    self.connectivity_tol,
                )

            solver.set_min_objective(  # type: ignore
                lambda x, g: self._nlopt_objective_f(x, g, settings)  # pyright: ignore[reportUnknownArgumentType]
            )

            for k, v in self.nlopt_params:
                solver.set_param(k, v)

            weights[:] = solver.optimize(weights)

            settings.update_weights(settings.vector_to_arrays(weights))
            logger.info(f"last index: {self.last_completed_index} i: {i}")

        optimal_design_weights = settings.filter_and_project(optimization=self)

        save_output(settings, self, 0, binarize=True)

        return optimal_design_weights
