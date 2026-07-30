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

from ..core.defs import ObjectiveFunction, ObjectiveReturn, WeightsType, RawWeightsType
from ..core.base_settings import (
    SimulationSettings,
    OptimizationSettings,
    register_objective,
)
from ..core.constraints import tensor_jacobian_product  # type: ignore
from ..utils.util import save_output

import numpy as np
import numpy.typing as npt
import nlopt  # type: ignore
from pydantic import Field, model_validator
from typing import Any


class NloptEpigraphOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `nlopt` backend for non-differentiable/multi-objective optimizations. Uses
    the epigraph formulation to recast the problem in a differentiable manner.
    (see: https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/#equivalent-formulations-of-optimization-problems)
    """

    linewidth_tol: float = 1e-3
    connectivity_tol: float = 1e-3
    optimizer: nlopt.opt = Field(exclude=True)
    n_objectives: int
    objective_constraints: npt.NDArray[np.float64] = np.array([])
    epigraph_min: float = 0
    epigraph_max: float = 1

    @model_validator(mode="before")
    @classmethod
    def fix_objective_constraints(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cons = data.get("objective_constraints")  # type: ignore
            n = data.get["n_objectives"]  # type: ignore

            if isinstance(cons, (int, float)) and n > 0:
                data["objectove_constraints"] = np.array([cons] * n, dtype=np.float64)
            elif isinstance(cons, list) and n > 0:
                if len(cons) == 0:  # type: ignore
                    # default of 1e-6 if not supplied
                    data["objectove_constraints"] = np.array(
                        [1e-6] * n, dtype=np.float64
                    )
                else:
                    data["objectove_constraints"] = np.array(cons, dtype=np.float64)

        return data  # type: ignore

    @model_validator(mode="after")
    def validate_n_objectives(self) -> "NloptEpigraphOptimizationSettings":
        if len(self.objective_constraints) != self.n_objectives:
            raise ValueError(
                f"Mismatch: n_objectives ({self.n_objectives}) "
                f"must match number of constraint values (len {len(self.objective_constraints)})"
            )

        return self

    def reorganize_grads(
        self, grads: npt.NDArray[np.float64], settings: SimulationSettings
    ) -> npt.NDArray[np.float64]:
        """Expects that gradients are returned from the optimization in the following
        order: [Obj0[Region0, Region1,...], Obj1[...],...] (but not in nested lists,
        in a huge vector).

        Args:
            grads (npt.NDArray[np.float64]): Input mega-list of gradients

        Returns:
            npt.NDArray[np.float64]: gradients sorted into a 2D array where each
            column corresponds to a different objective. (use like: grad[:, i] = ...)
        """

        chunk_size = settings.total_n()
        ret = np.zeros([chunk_size, self.n_objectives])
        start = 0
        stop = 0
        for i in range(self.n_objectives):
            stop = start + chunk_size
            ret[:, i] = grads[start:stop]
            start = stop + 1

        return ret

    def _linewidth_cons(
        self,
        epigrah_and_weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettings,
    ) -> float:

        weights = epigrah_and_weights[1:]
        settings.update_weights(settings.vector_to_arrays(weights))
        settings.apply_masks()
        settings.apply_symmetry()

        spacing, grad_out = settings.line_width_and_spacing(optimization=self)

        if grad_out.size > 0:
            # a1 hyperparameter from the constraint
            grad[:, 0] = -1e3
            grad[:, 1] = grad_out

        return spacing

    def _connectivity_cons(
        self,
        epigraph_and_weights: RawWeightsType,
        grad: RawWeightsType,
        settings: SimulationSettings,
    ) -> float:
        weights = epigraph_and_weights[1:]

        settings.update_weights(settings.vector_to_arrays(weights))
        settings.apply_masks()
        settings.apply_symmetry()

        connectivity, grad_out = settings.connectivity_constraint(optimization=self)
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
        settings: SimulationSettings,
    ) -> float:

        epigraph = epigraph_and_weights[0]
        weights = epigraph_and_weights[1:]

        opt = settings.get_objective(self)

        f0, grad = opt(weights)

        if gradient.size > 0:
            gradient[:, 0] = -1  # gradient with respect to epigraph variable
            gradient[:, 1:] = grad[0][0].T  # real gradients per-parameter

        result[:] = np.real(f0) - epigraph

        obj: float = np.linalg.norm(f0)  # type: ignore
        self.obj.append(np.real(obj))
        self.weights.append(epigraph_and_weights[1:].copy())
        self.data.append(epigraph)

        return obj

    def optimize(self, settings: SimulationSettings) -> list[WeightsType]:
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
        weights = np.concatenate(settings.weights_raw)

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
                    lambda x, g: self._linewidth_cons(x, g, settings),
                    self.linewidth_tol,
                )

            if self.apply_connectivity:
                solver.add_inequality_constraint(  # type: ignore
                    lambda x, g: self._connectivity_cons(x, g, settings),
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
                mconstraint, self.objective_constraints
            )

            weights_and_epigraph[:] = solver.optimize(weights_and_epigraph)  # type: ignore

            save_output(weights_and_epigraph, settings, self, sigmoid_bias)
            self.last_completed_index = i

        optimal_design_weights = settings.filter_and_project(
            weights=weights[1:], optimization=self
        )

        save_output(weights_and_epigraph[1:], settings, self, 0, binarize=True)

        return optimal_design_weights


@register_objective(SimulationSettings, NloptEpigraphOptimizationSettings)
def _(
    settings: SimulationSettings, optimization: NloptEpigraphOptimizationSettings
) -> ObjectiveFunction:
    def get_nlopt_epigraph_objective(
        weights: RawWeightsType,
    ) -> ObjectiveReturn:
        """Default objective function used for `nlopt` optimizations. Best used
        for multi-objective situations.

        Expects that gradients are returned from the optimization in the following
        order: [Obj0[Region0, Region1,...], Obj1[...],...] (but not in nested lists,
        in a huge vector).

        Args:
            weights (RawWeightsType): List of weights concatenated together.

        Returns:
            out (ObjectiveReturn): FOM and gradients for the given weights
        """

        opt = settings.create_opt(optimization)

        obj_val, dJ_du = opt(
            settings.filter_and_project(
                weights=settings.raw_to_weightslike(weights), optimization=optimization
            )
        )

        organized = optimization.reorganize_grads(grads=dJ_du, settings=settings)

        grad = np.zeros((settings.total_n(), optimization.n_objectives))
        for k in range(optimization.n_objectives):
            # grad[:, k] = tensor_jacobian_product(settings.filter_and_project, 0)(
            #     weights,
            #     optimization,
            #     organized[:, k],
            # )

            grad[:, k] = tensor_jacobian_product(wrapped_filter_and_project, 0)(
                weights,
                settings,
                optimization,
                organized[:, k],
            )

        return obj_val, grad

    return get_nlopt_epigraph_objective
