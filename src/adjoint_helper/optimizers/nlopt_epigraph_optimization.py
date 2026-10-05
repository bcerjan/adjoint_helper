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

__all__ = ["NloptEpigraphOptimizationSettings"]

import logging
from typing import Any

import nlopt  # type: ignore
import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from adjoint_helper.core.pydantic_defs import SolverConfig

from ..core.base_settings import (
    OptimizationSettings,
    SimulationSettings,
    register_objective,
)
from ..core.defs import (
    ObjectiveFunction,
    ObjectiveReturn,
    RawWeightsType,
    WeightsType,
)
from ..utils.util import save_output

logger = logging.getLogger(__name__)

DEFAULT_OBJECTIVE_TOL = 1e-6


class NloptEpigraphOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `nlopt` backend for non-differentiable/multi-objective optimizations. Uses
    the epigraph formulation to recast the problem in a differentiable manner.
    (see: https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/#equivalent-formulations-of-optimization-problems)

    Note that 'objective_constraints' should be either a 3D or 1D array of constraints
    if you supply it directly. It should have size self.simulation_settings.n_design_regions x
    self.simulation_settings.n_objective_functions x self.simulation_settings.n_design_parameters. This can either
    be a 3D array or a 1D array that is of the appropriate length. If it is 1D
    it will be unflattened to the correct shape.

    If omitted, a default of 1e-6 will be used. If a single numer or 1-element array
    is supplied, that number will be used for all of the constraints.
    """

    linewidth_tol: float = 1e-3
    connectivity_tol: float = 1e-3
    optimizer: nlopt.opt = Field(exclude=True)
    n_objectives: int
    objective_constraints: RawWeightsType = np.array([])
    epigraph_min: float = -np.inf
    epigraph_max: float = 0

    @model_validator(mode="before")
    @classmethod
    def check_n_objectives(cls, data: Any) -> Any:
        if isinstance(data, dict):
            n: int = data.get("n_objectives")  # type: ignore
            if n <= 0:
                raise ValueError(f"n_objectives must be positive, got: {n}")

            settings = data.get("simulation_settings")

            if settings is None:
                raise ValueError("Must have settings passed in")

            if n != settings.n_objectives:
                raise ValueError(
                    f"The optimization settings and simulation settings must agree "
                    f"on the number of objectives to solve for. Optimization has: {n} "
                    f"and Simulation has: {settings.n_objectives}"
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def fix_objective_constraints(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cons = data.get("objective_constraints", np.array([]))  # type: ignore

            n: int = data.get("n_objectives")  # type: ignore

            if isinstance(cons, (int, float)) and n > 0:
                data["objective_constraints"] = np.array([cons] * n, dtype=np.float64)
            elif isinstance(cons, list) and n > 0:
                if len(cons) == 0:  # type: ignore
                    data["objective_constraints"] = np.array(
                        [DEFAULT_OBJECTIVE_TOL] * n, dtype=np.float64
                    )
                elif len(cons) == 1:  # type: ignore
                    data["objective_constraints"] = np.array(cons * n, dtype=np.float64)
                else:
                    data["objective_constraints"] = np.array(cons, dtype=np.float64)
            elif isinstance(cons, np.ndarray):
                if cons.ndim == 1 and cons.size == 1:
                    data["objective_constraints"] = np.full(n, cons, dtype=np.float64)
                elif cons.size == 0:
                    data["objective_constraints"] = np.array(
                        [DEFAULT_OBJECTIVE_TOL] * n, dtype=np.float64
                    )

        return data  # type: ignore

    @property
    def solver_config(self) -> SolverConfig:
        return super().solver_config

    def _stack_grads(self, grads: list[list[list[RawWeightsType]]]) -> RawWeightsType:
        """
        Args:
            grads (list[list[list[RawWeightsType]]]): Input mega-list of gradients in the
            standard return format from an optimization (`ObjectiveReturn`)

        Returns:
            RawWeightsType: gradients stacked into a single ginormous
            vector. Note that there will be more than one gradient for each
            design region as we need all of them later on, but we are required
            to return a 1D vector due to our type constraints.
        """

        ret = [
            parameter.ravel()
            for design_region in grads
            for objective_function in design_region
            for parameter in objective_function
        ]

        return np.concatenate(ret)

    def _unstack_grads(self, grads: RawWeightsType) -> list[list[list[RawWeightsType]]]:
        idx = 0
        ret = []

        for dr_i in range(self.simulation_settings.n_design_regions):
            chunk_size = self.simulation_settings.design_regions[dr_i].total_n
            drs = []
            for _ in range(self.simulation_settings.n_objective_functions):
                objs = []
                for _ in range(self.simulation_settings.n_design_parameters):
                    objs.append(grads[idx : idx + chunk_size])
                    idx += chunk_size

                drs.append(objs)
            ret.append(drs)

        return ret

    def _stack_design_region_grads(
        self, grads: list[list[list[RawWeightsType]]]
    ) -> list[list[RawWeightsType]]:
        """Reorganizes the gradients so they are in a ~3D matrix (list of list of 1d vector)
        for the shape the nlopt expects them to be in. This means that the first  dimension
        is objective_function_idx, second is n_design_parameter_idx, third is stacked gradients
        for all design regions."""
        temp = [
            [np.array([])] * self.simulation_settings.n_design_parameters
        ] * self.simulation_settings.n_objective_functions
        for obj_idx in range(self.simulation_settings.n_objective_functions):
            for param_idx in range(self.simulation_settings.n_design_parameters):
                temp_gs = []
                for dr_idx in range(self.simulation_settings.n_design_regions):
                    temp_gs.append(grads[dr_idx][obj_idx][param_idx])
                temp[obj_idx][param_idx] = np.concatenate(temp_gs)
        return temp

    def _linewidth_cons(
        self,
        epigrah_and_weights: RawWeightsType,
        grad: RawWeightsType,
    ) -> float:

        weights = epigrah_and_weights[1:]
        self.simulation_settings.update_weights(
            self.simulation_settings.vector_to_arrays(weights)
        )
        self.simulation_settings.apply_masks()
        self.simulation_settings.apply_symmetry()

        spacing, grad_out = self.simulation_settings.line_width_and_spacing(
            optimization=self
        )

        if grad_out.size > 0:
            # a1 hyperparameter from the constraint
            grad[:, 0] = -1e3
            grad[:, 1] = grad_out

        return spacing

    def _connectivity_cons(
        self,
        epigraph_and_weights: RawWeightsType,
        grad: RawWeightsType,
    ) -> float:
        weights = epigraph_and_weights[1:]

        self.simulation_settings.update_weights(
            self.simulation_settings.vector_to_arrays(weights)
        )
        self.simulation_settings.apply_masks()
        self.simulation_settings.apply_symmetry()

        connectivity, grad_out = self.simulation_settings.connectivity_constraint(
            optimization=self
        )
        if grad_out.size > 0:
            # TODO: What should the grad be here?
            grad[:, 0] = -1e-3
            grad[:, 1] = grad_out

        return connectivity

    def _nlopt_epigraph_constraint(
        self,
        result: npt.NDArray[np.float64],
        epigraph_and_weights: RawWeightsType,
        gradient: RawWeightsType,
    ) -> float:

        epigraph = epigraph_and_weights[0]
        weights = epigraph_and_weights[1:]

        opt = self.simulation_settings.get_objective(self)

        f0, grad = opt(weights)
        grad = self._unstack_grads(grads=grad)
        grad = self._stack_design_region_grads(grads=grad)
        grad = np.array(grad)
        grad.reshape(self.simulation_settings.n_objectives, -1)

        if gradient.size > 0:
            gradient[:, 0] = -1  # gradient with respect to epigraph variable
            gradient[:, 1:] = grad  # real gradients per-parameter

        result[:] = np.real(f0) - epigraph

        obj: float = np.linalg.norm(f0)  # type: ignore
        self.fom.append(np.real(obj))
        self.weights.append(self.simulation_settings.vector_to_arrays(weights.copy()))
        self.data.append(epigraph)

        return obj

    def _nlopt_epigraph_f(
        self, epigraph_and_weights: RawWeightsType, grad: RawWeightsType
    ) -> float:
        t = epigraph_and_weights[0]

        if grad.size > 0 and grad is not None:
            grad[0] = 1
            grad[1:] = 0

        return t

    def optimize(self) -> list[WeightsType]:
        """Runs an nlopt optimization process using the epigraph formulation.
        Use this version if you have "fancy" things going on -- multiple design
        objectives, multiple design parameters, etc. Multiple design regions
        can be handled by any of the optimizers

        Args:
            settings (SimulationSettings): Simulation settings for this simulation
            optimization (OptimizationSettings): Optimization settings

        Returns:
            npt.NDArray[np.float64]: Optimal weights after the optimization
        """

        # First, make sure n_objectives and the settings object agree:
        if self.n_objectives != self.simulation_settings.n_objectives:
            raise ValueError(
                f"The optimization settings and simulation settings must agree "
                f"on the number of objectives to solve for. Optimization has: {self.n_objectives} "
                f"and Simulation has: {self.simulation_settings.n_objectives}"
            )

        self.simulation_settings.calculate_normalization()

        lb = np.zeros(self.simulation_settings.total_n())
        ub = np.ones(self.simulation_settings.total_n())
        weights = np.concatenate(self.simulation_settings.weights_raw)

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            weights = self.weights[-1]

        weights_and_epigraph = np.insert(weights, 0, -1)  # initial guess

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
                    self._linewidth_cons,
                    self.linewidth_tol,
                )

            if self.apply_connectivity:
                solver.add_inequality_constraint(  # type: ignore
                    self._connectivity_cons,
                    self.connectivity_tol,
                )

            def mconstraint(
                result: RawWeightsType,
                epigraph_and_weights: RawWeightsType,
                gradient: RawWeightsType,
            ):
                self._nlopt_epigraph_constraint(
                    result=result,
                    epigraph_and_weights=epigraph_and_weights,
                    gradient=gradient,
                )

            solver.add_inequality_mconstraint(  # type: ignore
                mconstraint, self.objective_constraints.flatten()
            )
            # Should be gradients for all design regions as a function
            # of obj. func. and design param. so a 3D system of size
            # n_objs x n_params x total_n
            solver.set_min_objective(self._nlopt_epigraph_f)

            weights_and_epigraph[:] = solver.optimize(weights_and_epigraph)  # type: ignore

            save_output(optimization=self, sigmoid_bias=sigmoid_bias)

        optimal_design_weights = self.simulation_settings.filter_and_project(
            optimization=self
        )

        save_output(optimization=self, sigmoid_bias=0, binarize=True)

        return optimal_design_weights


@register_objective(SimulationSettings, NloptEpigraphOptimizationSettings)
def _(
    settings: SimulationSettings, optimization: NloptEpigraphOptimizationSettings
) -> ObjectiveFunction:
    def get_nlopt_epigraph_objective(
        weights: RawWeightsType,
    ) -> ObjectiveReturn:
        """Default objective function used for `nlopt` epigraph optimizations. Best used
        for multi-objective situations.

        If you re-implement something like this yourself, you _must_ call
        ret = optimization._stack_grads(grads) and then return `ret` as the
        gradient. This stacks all the grads into a single giant vector
        which is what the rest of the code expects. It will be unstacked later
        on internally.

        Args:
            weights (RawWeightsType): List of weights concatenated together.

        Returns:
            out (ObjectiveReturn): FOM and gradients for the given weights.
            For this objective specifically, there is more than one gradient
            per design region.
        """
        settings.update_weights(settings.vector_to_arrays(weights))
        settings.apply_masks()
        settings.apply_symmetry()

        opt = settings.create_opt(optimization)

        obj_val, dJ_du = opt(settings.filter_and_project(optimization=optimization))

        temp_grads = settings.get_empty_grads()

        for dr_i, obj_func, param, r in settings.design_space:
            r.backpropagate(dJ_du[dr_i][obj_func][param], optimization=optimization)
            temp_grads[dr_i][obj_func][param][:] = dJ_du[dr_i][obj_func][param]

        logger.info(
            f"iteration: {len(optimization.data) + 1}, "
            f"sigmoid_bias: {optimization.sigmoid_bias}, "
            f"obj. value: {obj_val}"
        )

        grad = optimization._stack_grads(temp_grads)

        return obj_val, grad

    return get_nlopt_epigraph_objective
