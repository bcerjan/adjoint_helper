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

import numpy as np
import numpy.typing as npt
import optax  # type: ignore
from typing import Tuple
from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.simulation_settings import SimulationSettings, Optimization_Func
from adjoint_helper.util import save_output

from adjoint_helper.constraints import (
    filter_and_project,
    connectivity_constraint,
    line_width_and_spacing_constraint,  # type: ignore
    tensor_jacobian_product,  # type: ignore
)


class OptaxOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `optax` backend.
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

    def obj_func(
        self,
        weights: npt.NDArray[np.float64],
        opt: Optimization_Func,
        settings: SimulationSettings,
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """Constraint function for the epigraph formulation.

        Args:
        weights: 1D array containing design weights.
        opt: Meep Adjoint Optimization Problem
        settings: SimulationSettings containing the geometrical parameters
        optimization: OptimizationSettings containing optimization parameters
        """

        obj_val, grad = opt([filter_and_project(weights, settings, self)])  # type: ignore

        obj: float = obj_val[0]  # type: ignore

        grad[:] = tensor_jacobian_product(filter_and_project, 0)(
            weights,
            settings,
            self,
            grad,
        )

        print(
            f"iteration: {len(self.data)}, sigmoid_bias: {self.sigmoid_bias}, "
            f"obj. func.: {obj}, "
        )

        # Apply penalties:
        if self.apply_connectivity:
            connectivity, g = connectivity_constraint(
                weights=weights,
                settings=settings,
                optimization=self,
            )

            if connectivity > 0:
                grad[:] = grad[:] + g[:] * self.connectivity_penalty
                obj += connectivity * self.connectivity_penalty

        if self.apply_linewidth:
            fabrication: float
            fabrication, g = line_width_and_spacing_constraint(  # type: ignore
                weights=weights,
                gradient=grad,
                settings=settings,
                optimization=self,
            )  # type: ignore

            if fabrication > 0:
                grad[:] = grad[:] + g[:] * self.connectivity_penalty
                obj += fabrication * self.connectivity_penalty  # type: ignore

        self.obj.append(np.real(obj))  # type: ignore
        self.weights.append(weights.copy())

        return obj, grad  # type: ignore

    def optimize(self, settings: SimulationSettings) -> npt.NDArray[np.float64]:
        """
        Runs the optimization and stores results. Returns the (projected) optimal
        weights for external stuff if desired
        """
        masks = settings.border_masks(self)

        num_weights = settings.total_n()

        # Initial design weights (arbitrary constant value).
        weights = np.ones((num_weights,)) * 0.5
        # weights = np.random.rand(num_weights)

        for mask in masks:
            weights[mask.locations.flatten()] = mask.value

        optimal_design_weights = np.zeros_like(weights)

        history_fpath = settings.data_dir + settings.history_fname

        self.use_damping = True

        # Handle restarting:
        last_index = self.last_completed_index
        biases = self.sigmoid_biases

        if last_index >= 0:
            weights = self.weights[-1]

        learning_rate = 0.75

        optimizer = optax.adam(learning_rate=learning_rate)  # type: ignore
        opt_state = optimizer.init(weights)  # type: ignore

        for idx, sigmoid_bias in enumerate(
            biases[last_index + 1 :], start=last_index + 1
        ):
            max_eval = self.max_evals[idx]

            self.apply_settings(sigmoid_bias)

            opt = settings.create_opt(self)

            for i in range(max_eval):
                val, grad = self.obj_func(weights, opt, settings)

                updates, opt_state = optimizer.update(grad, opt_state, weights)  # type: ignore

                weights[:] = optax.apply_updates(weights, updates)  # type: ignore

                weights[:] = np.clip(weights, 0.0, 1.0)
                for mask in masks:
                    weights[mask.locations.flatten()] = mask.value

                # outputs
                print(f"\nstep = {i + 1}")
                print(f"\tobjective = {val:.4e}")
                print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}\n")

            save_output(weights, settings, self, sigmoid_bias, history_fpath)

            self.last_completed_index = idx

        optimal_design_weights = filter_and_project(
            weights[:],
            settings,
            self,
        )

        save_output(weights, settings, self, 0, history_fpath, binarize=True)

        return optimal_design_weights
