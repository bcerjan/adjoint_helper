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

from ..core.defs import PhysicsObjective, ObjectiveReturn, WeightsType, RawWeightsType
from ..core.export_settings import OptimizationSettings
from ..core.base_settings import SimulationSettingsBase, register_physics_objective
from ..core.constraints import tensor_jacobian_product  # type: ignore
from ..utils.util import save_output, apply_masks, wrapped_filter_and_project

import numpy as np
import optax  # type: ignore
from pydantic import Field, model_validator
from typing import Any, Annotated


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

    def optimize(self, settings: SimulationSettingsBase) -> WeightsType:
        """
        Runs the optimization and stores results. Returns the (projected) optimal
        weights for external stuff if desired
        """

        settings.calculate_normalization()

        num_weights = settings.total_n_raw()

        # Initial design weights (arbitrary constant value).
        weights = np.ones((num_weights,)) * 0.5

        apply_masks(
            masks=settings.get_masks(self.filter_radius),
            weights=weights,
            multi_region=settings.is_multi_region(),
        )

        optimal_design_weights = np.zeros_like(weights)

        self.use_damping = True

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            weights = self.weights[-1]

        optimizer = self.optimizer  # type: ignore
        opt_state = optimizer.init(weights)  # type: ignore

        for idx, sigmoid_bias in enumerate(
            biases[last_index + 1 :], start=last_index + 1
        ):
            max_eval = self.max_evals[idx]

            self.sigmoid_bias = sigmoid_bias

            opt = settings.create_opt(self)

            for i in range(max_eval):
                val, grad = opt(weights)

                updates, opt_state = optimizer.update(grad, opt_state, weights)  # type: ignore

                weights[:] = optax.apply_updates(weights, updates)  # type: ignore

                weights[:] = np.clip(weights, 0.0, 1.0)

                apply_masks(
                    weights=weights,
                    masks=settings.get_masks(self.filter_radius),
                    multi_region=settings.is_multi_region(),
                )

                # outputs
                print(f"\nstep = {i + 1}")
                print(f"\tobjective = {np.linalg.norm(np.real(val)):.4e}")
                print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}\n")

            save_output(weights, settings, self, sigmoid_bias)

            self.last_completed_index = idx

        optimal_design_weights = settings.filter_and_project(
            weights=weights, optimization=self
        )

        save_output(weights, settings, self, 0, binarize=True)

        return settings.raw_to_weightslike(optimal_design_weights)


@register_physics_objective(SimulationSettingsBase, OptaxOptimizationSettings)
def _(
    settings: SimulationSettingsBase, optimization: OptaxOptimizationSettings
) -> PhysicsObjective:
    def get_optax_objective(
        weights: RawWeightsType,
    ) -> ObjectiveReturn:
        """Default objective function used for `optax` optimizations. For simple
        (single-objective) optimizations, this is sufficient as-is. For
        more complicated objectives, you will need to customize it for your
        needs. Weights should not be updated in-place.

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

        obj: float = np.sum(obj_val)

        grad[:] = tensor_jacobian_product(wrapped_filter_and_project, 0)(
            weights,
            settings,
            optimization,
            grad,
        )

        print(
            f"iteration: {len(optimization.data)}, sigmoid_bias: {optimization.sigmoid_bias}, "
            f"obj. func.: {obj}, "
        )

        # Apply penalties:
        if optimization.apply_connectivity:
            connectivity, g = settings.connectivity_constraint(
                weights=weights,
                optimization=optimization,
            )

            if connectivity > 0:
                grad[:] = grad[:] + g[:] * optimization.connectivity_penalty
                obj += connectivity * optimization.connectivity_penalty

        if optimization.apply_linewidth:
            fabrication: float
            fabrication, g = settings.line_width_and_spacing(
                weights=weights,
                optimization=optimization,
            )

            if fabrication > 0:
                grad[:] = grad[:] + g[:] * optimization.connectivity_penalty
                obj += fabrication * optimization.connectivity_penalty  # type: ignore

        optimization.obj.append(np.real(obj))  # type: ignore
        optimization.weights.append(weights.copy())

        return obj_val, grad

    return get_optax_objective
