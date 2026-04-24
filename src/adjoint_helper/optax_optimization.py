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
import meep.adjoint as mpa  # pyright: ignore[reportMissingTypeStubs]
import optax  # pyright: ignore[reportMissingTypeStubs]
from typing import Tuple
from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.util import save_output

# from pathlib import Path

from adjoint_helper.constraints import (
    filter_and_project,
    connectivity_constraint,
    line_width_and_spacing_constraint,  # pyright: ignore[reportUnknownVariableType]
    tensor_jacobian_product,  # pyright: ignore[reportUnknownVariableType]
)


def obj_func(
    weights: npt.NDArray[np.float_],
    opt: mpa.OptimizationProblem,
    settings: SimulationSettings,
    optimization: OptimizationSettings,
) -> Tuple[float, npt.NDArray[np.float_]]:
    """Constraint function for the epigraph formulation.

    Args:
      weights: 1D array containing design weights.
      opt: Meep Adjoint Optimization Problem
      settings: SimulationSettings containing the geometrical parameters
      optimization: OptimizationSettings containing optimization parameters
    """

    obj_val, grad = opt([filter_and_project(weights, settings, optimization)])  # type: ignore

    obj: float = obj_val[0]  # type: ignore

    grad[:] = tensor_jacobian_product(filter_and_project, 0)(
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
        connectivity, g = connectivity_constraint(
            weights=weights,
            settings=settings,
            optimization=optimization,
        )

        if connectivity > 0:
            grad[:] = grad[:] + g[:] * optimization.penalty_weight
            obj += connectivity * optimization.penalty_weight

    if optimization.apply_linewidth:
        fabrication: float
        fabrication, g = line_width_and_spacing_constraint(  # type: ignore
            weights=weights,
            gradient=grad,
            settings=settings,
            optimization=optimization,
        )  # type: ignore

        if fabrication > 0:
            grad[:] = grad[:] + g[:] * optimization.penalty_weight
            obj += fabrication * optimization.penalty_weight  # type: ignore

    optimization.obj.append(np.real(obj))  # type: ignore
    optimization.weights.append(weights.copy())

    return obj, grad  # type: ignore


def run_adam_optimization(
    settings: SimulationSettings, optimization: OptimizationSettings
) -> np.ndarray[(int), np.dtype[np.float_]]:
    """
    Runs the optimization and stores results. Returns the (projected) optimal
    weights for external stuff if desired
    """
    masks = settings.border_masks(optimization)

    num_weights = settings.total_n()

    # Initial design weights (arbitrary constant value).
    weights = np.ones((num_weights,)) * 0.5
    # weights = np.random.rand(num_weights)

    for mask in masks:
        weights[mask.locations.flatten()] = mask.value

    optimal_design_weights = np.zeros_like(weights)

    history_fpath = settings.data_dir + settings.history_fname

    optimization.use_damping = True

    # Handle restarting:
    last_index = optimization.last_completed_index
    biases = optimization.sigmoid_biases

    if last_index >= 0:
        weights = optimization.weights[-1]

    learning_rate = 0.75

    optimizer = optax.adam(learning_rate=learning_rate)  # type: ignore
    opt_state = optimizer.init(weights)  # type: ignore

    for idx, sigmoid_bias in enumerate(biases[last_index + 1 :], start=last_index + 1):
        max_eval = optimization.max_evals[idx]

        optimization.apply_settings(sigmoid_bias)

        opt = settings.create_opt(optimization)

        for i in range(max_eval):
            val, grad = obj_func(weights, opt, settings, optimization)

            updates, opt_state = optimizer.update(grad, opt_state, weights)  # type: ignore

            weights[:] = optax.apply_updates(weights, updates)  # type: ignore

            weights[:] = np.clip(weights, 0.0, 1.0)
            for mask in masks:
                weights[mask.locations.flatten()] = mask.value

            # outputs
            print(f"\nstep = {i + 1}")
            print(f"\tobjective = {val:.4e}")
            print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}\n")

        save_output(weights, settings, optimization, sigmoid_bias, history_fpath)

        optimization.last_completed_index = idx

    optimal_design_weights = filter_and_project(
        weights[:],
        settings,
        optimization,
    )

    save_output(weights, settings, optimization, 0, history_fpath, binarize=True)

    return optimal_design_weights
