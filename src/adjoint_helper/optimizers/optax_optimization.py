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

from adjoint_helper.core.pydantic_defs import SolverConfig
from ..core.defs import (
    ObjectiveReturn,
    WeightsType,
    RawWeightsType,
)
from ..core.base_settings import (
    SimulationSettings,
    OptimizationSettings,
)
from ..utils.util import save_fom_history, save_output

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
    optimizer: Annotated[Any, Field(exclude=True, repr=False)] = optax.adam(
        learning_rate=0.2
    )
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

    @property
    def solver_config(self) -> SolverConfig:
        return super().solver_config

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
        save_output(
            settings=settings, optimization=self, sigmoid_bias=self.sigmoid_bias
        )

        return obj, grad

    def optimize(self, settings: SimulationSettings) -> list[WeightsType]:
        """
        Runs the optimization and stores results. Returns the (projected) optimal
        weights for external stuff if desired
        """

        settings.calculate_normalization()

        # Initial design weights (arbitrary constant value).
        weights: RawWeightsType = np.concatenate(settings.weights_raw)

        settings.apply_masks()

        optimal_design_weights = np.zeros_like(weights)

        # Handle restarting:
        _, last_index = self._current_bias_and_index()
        biases = self.sigmoid_biases

        optimizer = self.optimizer
        opt_state = optimizer.init(weights)

        for idx, sigmoid_bias in enumerate(biases[last_index:], start=last_index):
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

        optimal_design_weights = settings.filter_and_project(optimization=self)

        save_output(settings, self, 0, binarize=True)
        save_fom_history(optimization=self, history_dir=settings.data_dir)

        return optimal_design_weights
