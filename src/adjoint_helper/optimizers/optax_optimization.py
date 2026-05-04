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

from ..core.defs import PhysicsObjective, ObjectiveReturn, WeightsType, RawWeightsType
from ..core.export_settings import OptimizationSettings
from ..core.base_settings import SimulationSettingsBase
from ..core.constraints import tensor_jacobian_product  # type: ignore
from ..core.objective_factory import get_physics_objective
from ..utils.util import save_output, apply_masks

import numpy as np
import optax  # type: ignore


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

    connectivity_penalty: float
    linewidth_penalty: float
    optmizer: optax.GradientTransformationExtraArgs

    def __init__(
        self,
        minimum_size: float = 0.05,
        sigmoid_bias_threshold: float = 32,  # Sigmoid bias at which eps_avg turns on
        sigmoid_threshold: float = 0.5,  # Eta
        sigmoid_erosion: float = 0.65,  # Eta_e
        sigmoid_bias_init: float = 4,
        sigmoid_bias_scale: float = 1.2,
        connectivity_sigmoid_threshold: float = 16,
        linewidth_sigmoid_threshold: float = 24,  # Sigmoid bias at which line width constraint turns on
        total_evals: int = 40,
        maximum_runtime: float = 200,
        minimum_runtime: float = 0,
        decay_by: float = 1e-6,
        use_smoothed_projection: bool = False,
        do_connectivity: bool = False,
        connectivity_penalty: float = 0.2,
        linewidth_penalty: float = 0.2,
        optmizer: optax.GradientTransformationExtraArgs = optax.adam(learning_rate=0.2),
    ):
        self.connectivity_penalty = connectivity_penalty
        self.linewidth_penalty = linewidth_penalty
        self.optmizer = optmizer

        sigmoid_biases = [
            sigmoid_bias_init * sigmoid_bias_scale**i for i in range(total_evals)
        ]

        max_evals = np.ones(total_evals, dtype=np.int32).tolist()

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

    def optimize(self, settings: SimulationSettingsBase) -> WeightsType:
        """
        Runs the optimization and stores results. Returns the (projected) optimal
        weights for external stuff if desired
        """

        num_weights = settings.total_n_raw()

        # Initial design weights (arbitrary constant value).
        weights = np.ones((num_weights,)) * 0.5

        apply_masks(masks=settings.get_masks(self.filter_radius), weights=weights)

        optimal_design_weights = np.zeros_like(weights)

        history_fpath = settings.data_dir / settings.history_fname

        self.use_damping = True

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            weights = self.weights[-1]

        optimizer = self.optmizer  # type: ignore
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
                    weights=weights, masks=settings.get_masks(self.filter_radius)
                )

                # outputs
                print(f"\nstep = {i + 1}")
                print(f"\tobjective = {np.linalg.norm(np.real(val)):.4e}")
                print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}\n")

            save_output(weights, settings, self, sigmoid_bias, history_fpath)

            self.last_completed_index = idx

        optimal_design_weights = settings.filter_and_project(
            weights=weights, optimization=self
        )

        save_output(weights, settings, self, 0, history_fpath, binarize=True)

        return settings.raw_to_weightslike(optimal_design_weights)


@get_physics_objective.register
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
            settings.filter_and_project(weights=weights, optimization=optimization)
        )  # type: ignore

        obj: float = np.sum(obj_val)

        grad[:] = tensor_jacobian_product(settings.filter_and_project, 0)(
            weights,
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
