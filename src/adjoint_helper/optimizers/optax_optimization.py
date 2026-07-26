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

__all__ = ["OptaxOptimizationSettings"]

from ..core.defs import (
    ObjectiveReturn,
    WeightsType,
    RawWeightsType,
)
from ..core.base_settings import (
    SimulationSettings,
    OptimizationSettings,
)
from ..utils.util import save_output

import logging
import numpy as np
import optax  # type: ignore
from pydantic import Field, model_validator
from typing import Any, Annotated

logger = logging.getLogger(__name__)


class OptaxOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `optax` backend. Passes the list of gradients and list of objective values
    directly -- this might not work for all optimizers, so choose carefully. If
    you want a better-controlled approach to multiobjective stuff, you can switch
    to the `NloptEpigraphOptimizationSettings` optimizer which uses the epigraph
    formulation (https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/#equivalent-formulations-of-optimization-problems)

    Assumes there is only a single set of weights / gradients. Most (all?) of the
    `optax` optimizers work this way. If you have a more complicated objective function
    you have two choices -- either use a different backend (e.g.
    `NloptEpigraphOptimizationSettings`) or override `get_physics_objective`
    and have it combine everything however you want.
    """

    connectivity_penalty: float = 0.2
    linewidth_penalty: float = 0.2
    # optimizer: optax.GradientTransformationExtraArgs = Field(exclude=True)  # type:
    optimizer: Annotated[Any, Field(exclude=True)] = optax.adam(learning_rate=0.2)
    sigmoid_bias_init: float = 4
    sigmoid_bias_scale: float = 1.2
    total_evals: int = 40

    @model_validator(mode="before")
    @classmethod
    def calculate_biases_and_evals(cls, data: Any) -> Any:
        # If we update defaults above, need to change here as well
        if isinstance(data, dict):
            init = data.get("sigmoid_bias_init", 4)  # type: ignore
            scale = data.get("sigmoid_bias_scale", 1.2)  # type: ignore
            n = data.get("total_evals", 40)  # type: ignore

            if isinstance(n, int):
                data["max_evals"] = np.ones(n, dtype=np.int32).tolist()

                if isinstance(init, (float, int)) and isinstance(scale, (float, int)):
                    data["sigmoid_biases"] = [init * scale**i for i in range(n)]

        return data  # type: ignore

    def _optax_f(
        self, settings: SimulationSettings, weights: RawWeightsType
    ) -> ObjectiveReturn:

        opt = settings.get_objective(self)
        obj_val, grad = opt(weights)
        obj = np.sum(np.real(obj_val))
        if self.apply_connectivity:
            connectivity, g = settings.connectivity_constraint(optimization=self)

            if connectivity > 0:
                grad[:] = grad[:] + g[:] * self.connectivity_penalty
                obj += connectivity * self.connectivity_penalty

        if self.apply_linewidth:
            fabrication: float
            fabrication, g = settings.line_width_and_spacing(optimization=self)

            if fabrication > 0:
                grad[:] = grad[:] + g[:] * self.linewidth_penalty
                obj += fabrication * self.linewidth_penalty

        self.obj.append(obj)
        self.weights.append(settings.vector_to_arrays(weights))
        self.data.append(obj)

        return obj, grad

    def optimize(self, settings: SimulationSettings) -> list[WeightsType]:
        """
        Runs the optimization and stores results. Returns the (projected) optimal
        weights for external stuff if desired
        """

        settings.calculate_normalization()

        num_weights = settings.total_n()

        # Initial design weights (arbitrary constant value).
        weights = np.ones((num_weights,)) * 0.5

        settings.apply_masks()

        optimal_design_weights = np.zeros_like(weights)

        self.use_damping = True

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            last_weights = self.weights[-1]
            weights = settings.arrays_to_vector(last_weights)

        optimizer = self.optimizer
        opt_state = optimizer.init(weights)

        for idx, sigmoid_bias in enumerate(
            biases[last_index + 1 :], start=last_index + 1
        ):
            max_eval = self.max_evals[idx]

            self.sigmoid_bias = sigmoid_bias

            for _ in range(max_eval):
                _, grad = self._optax_f(
                    settings=settings,
                    weights=weights,
                )

                updates, opt_state = optimizer.update(grad, opt_state, weights)

                weights[:] = optax.apply_updates(weights, updates)  # type: ignore

                settings.apply_masks()
                settings.apply_symmetry()
                weights[:] = np.clip(weights, 0.0, 1.0)

            save_output(settings=settings, optimization=self, sigmoid_bias=sigmoid_bias)

            self.last_completed_index = idx

        optimal_design_weights = settings.filter_and_project(optimization=self)

        save_output(settings=settings, optimization=self, sigmoid_bias=0, binarize=True)

        return optimal_design_weights


#
# @register_objective(SimulationSettings, OptaxOptimizationSettings)
# def _(
#     settings: SimulationSettings, optimization: OptaxOptimizationSettings
# ) -> ObjectiveFunction:
#     def get_optax_objective(
#         weights: RawWeightsType,
#     ) -> ObjectiveReturn:
#         """Default objective function used for `optax` optimizations. For simple
#         (single-objective) optimizations, this is sufficient as-is. For
#         more complicated objectives, you will need to customize it for your
#         needs. Weights should not be updated in-place.
#
#         Args:
#             weights (RawWeightsType): List of weights concatenated together.
#
#         Returns:
#             out (ObjectiveReturn): FOM and gradients for the given weights
#         """
#
#         settings.update_weights(settings.vector_to_arrays(weights))
#         settings.apply_masks()
#         settings.apply_symmetry()
#
#         opt = settings.create_opt(optimization)
#
#         obj_val, grads = opt(settings.filter_and_project(optimization=optimization))
#
#         grad_out = np.zeros((settings.total_n(),), dtype=np.float64)
#         temp_grads = settings.get_empty_grads()
#
#         for dr_i, r in enumerate(settings.design_regions):
#             for obj_func in range(settings.n_objective_functions):
#                 for param in range(settings.n_design_parameters):
#                     r.backpropagate(
#                         grads[dr_i][obj_func][param], optimization=optimization
#                     )
#                     temp_grads[dr_i][obj_func][param][:] = grads[dr_i][obj_func][param]
#
#         grad_out = combine_gradients(temp_grads, operation="sum")
#         grad = np.concatenate(grad_out)
#
#         obj: float = np.sum(np.real(obj_val))
#
#         logger.info(
#             f"iteration: {len(optimization.data)}, sigmoid_bias: {optimization.sigmoid_bias}, "
#             f"obj. func.: {obj}, "
#         )
#
#         # Apply penalties:
#         if optimization.apply_connectivity:
#             connectivity, g = settings.connectivity_constraint(
#                 optimization=optimization
#             )
#
#             if connectivity > 0:
#                 grad[:] = grad[:] + g[:] * optimization.connectivity_penalty
#                 obj += connectivity * optimization.connectivity_penalty
#
#         if optimization.apply_linewidth:
#             fabrication: float
#             fabrication, g = settings.line_width_and_spacing(optimization=optimization)
#
#             if fabrication > 0:
#                 grad[:] = grad[:] + g[:] * optimization.connectivity_penalty
#                 obj += fabrication * optimization.connectivity_penalty  # type: ignore
#
#         optimization.obj.append(np.real(obj))  # type: ignore
#         optimization.weights.append(settings.vector_to_arrays(weights))
#         if len(optimization.obj) > 1:
#             diff = np.abs(optimization.obj[-1] - optimization.obj[-2])
#             logger.info(f"\tobjective diff: {diff}")
#             if diff <= 1e-8:
#                 logger.error(
#                     "Objective did not change, something is wrong so we are aborting."
#                 )
#                 exit()
#
#         return obj_val, grad
#
#     return get_optax_objective
