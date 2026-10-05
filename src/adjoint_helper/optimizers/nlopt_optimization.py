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

import logging

import nlopt  # type: ignore
import numpy as np
from pydantic import Field, computed_field

from adjoint_helper.core.pydantic_defs import SolverConfig

from ..core.base_settings import (
    OptimizationSettings,
)
from ..core.defs import (
    RawWeightsType,
    WeightsType,
)
from ..utils.util import save_fom_history, save_output

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
    sigmoid_biases: list[float] = [4, 8, 16, 32, 64, 128]  # noqa: RUF012, pydantic handles this
    nlopt_params: dict = {}  # Can pass arbitrary settings through this to apply to the solver  # noqa: RUF012, pydantic handles

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
    ) -> float:

        self.simulation_settings.update_weights(
            self.simulation_settings.vector_to_arrays(weights)
        )
        self.simulation_settings.apply_masks()
        self.simulation_settings.apply_symmetry()

        spacing, grad_out = self.simulation_settings.line_width_and_spacing(
            optimization=self
        )

        if grad_out.size > 0:
            grad[:] = grad_out

        return spacing

    def _connectivity_cons(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
    ) -> float:

        self.simulation_settings.update_weights(
            self.simulation_settings.vector_to_arrays(weights)
        )
        self.simulation_settings.apply_masks()
        self.simulation_settings.apply_symmetry()

        connectivity, grad_out = self.simulation_settings.connectivity_constraint(
            optimization=self
        )
        if grad_out.size > 0:
            grad[:] = grad_out

        return connectivity

    def _nlopt_objective_f(
        self,
        weights: RawWeightsType,
        grad: RawWeightsType,
    ) -> float:
        """
        Internal objective function, just calls the user-supplied objective function
        and returns a value from it. Takes the norm of the real parts of the
        objective function value as uses that as the return.

        Needs to be a minimization objective

        Updates gradient in-place
        """

        opt = self.simulation_settings.get_objective(self)
        [obj], vec_out = opt(weights)

        if grad is not None and grad.size > 0:
            grad[:] = vec_out

        self.fom.append(obj)
        self.weights.append(self.simulation_settings.vector_to_arrays(weights.copy()))
        save_output(self, self.sigmoid_bias)

        return obj

    def optimize(self) -> list[WeightsType]:
        """Runs an nlopt optimization directly (no fancy business)

        Args:
            settings (SimulationSettings): Simulation settings for this simulation
            optimization (OptimizationSettings): Optimization settings

        Returns:
            npt.NDArray[np.float64]: Optimal weights after the optimization
        """

        self.simulation_settings.calculate_normalization()

        lb = np.zeros(self.simulation_settings.total_n())
        ub = np.ones(self.simulation_settings.total_n())
        weights: RawWeightsType = np.concatenate(self.simulation_settings.weights_raw)

        self.simulation_settings.apply_masks()

        # Handle restarting:
        _, last_index = self._current_bias_and_index()
        biases = self.sigmoid_biases

        for i, sigmoid_bias in enumerate(biases[last_index:], start=last_index):
            self.sigmoid_bias = sigmoid_bias
            solver = nlopt.opt(self.optimizer)
            solver.set_exceptions_enabled(True)
            solver.set_lower_bounds(lb)  # type: ignore
            solver.set_upper_bounds(ub)  # type: ignore

            solver.set_maxeval(self.max_evals[i])  # type: ignore

            if self.apply_linewidth:
                solver.add_inequality_constraint(  # type: ignore
                    self._linewidth_cons,
                    self.linewidth_tol,
                )

            if self.apply_connectivity:
                solver.add_inequality_constraint(  # type: ignore
                    self._connectivity_cons,
                    self.connectivity_tol,
                )

            solver.set_min_objective(  # type: ignore
                self._nlopt_objective_f
            )

            for k, v in self.nlopt_params:
                solver.set_param(k, v)

            weights[:] = solver.optimize(weights)

            self.simulation_settings.update_weights(
                self.simulation_settings.vector_to_arrays(weights)
            )

        optimal_design_weights = self.simulation_settings.filter_and_project(
            optimization=self
        )

        save_output(self, 0, binarize=True)
        save_fom_history(
            optimization=self, history_dir=self.simulation_settings.data_dir
        )

        return optimal_design_weights
