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

from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.optimization_settings import OptimizationSettings
import numpy as np
import numpy.typing as npt
from adjoint_helper.constraints import (
    line_width_and_spacing_constraint,
    connectivity_constraint,
    filter_and_project,
)
from adjoint_helper.util import save_output
import nlopt  # type: ignore


def _linewidth_cons(
    weights: npt.NDArray[np.float_],
    grad: npt.NDArray[np.float_],
    settings: SimulationSettings,
    optimization: OptimizationSettings,
) -> float:
    spacing, grad[:] = line_width_and_spacing_constraint(  # type: ignore
        weights=weights, gradient=grad, settings=settings, optimization=optimization
    )

    return spacing  # type: ignore


def _connectivity_cons(
    weights: npt.NDArray[np.float_],
    grad: npt.NDArray[np.float_],
    settings: SimulationSettings,
    optimization: OptimizationSettings,
) -> float:
    """_summary_

    Args:
        weights (npt.NDArray[np.float_]): _description_
        grad (npt.NDArray[np.float_]): _description_
        settings (SimulationSettings): _description_
        optimization (OptimizationSettings): _description_

    Returns:
        float: _description_
    """
    connectivity, grad[:] = connectivity_constraint(weights, settings, optimization)

    return connectivity


def run_nlopt_optimization(
    settings: SimulationSettings, optimization: OptimizationSettings
) -> npt.NDArray[np.float_]:
    """_summary_

    Args:
        settings (SimulationSettings): Simulation settings for this simulation
        optimization (OptimizationSettings): Optimization settings

    Returns:
        npt.NDArray[np.float_]: Optimal weights after the optimization
    """

    lb = np.zeros((settings.total_n(),))
    ub = np.ones((settings.total_n(),))
    weights = np.ones(settings.total_n()) * 0.5

    for mask in settings.border_masks(optimization):
        weights[mask.locations.flatten()] = mask.value

    history_fpath = settings.data_dir + settings.history_fname

    optimization.use_damping = True

    # Handle restarting:
    last_index = optimization.last_completed_index
    biases = optimization.sigmoid_biases

    if last_index >= 0:
        weights = optimization.weights[-1]

    for i, sigmoid_bias in enumerate(biases[last_index + 1 :], start=last_index + 1):
        optimization.apply_settings(sigmoid_bias)
        solver = nlopt.opt(nlopt.LD_CCSAQ, settings.total_n())  # type: ignore
        solver.set_lower_bounds(lb)  # type: ignore
        solver.set_upper_bounds(ub)  # type: ignore

        solver.set_maxeval(optimization.max_evals[i])  # type: ignore

        if optimization.apply_linewidth:
            solver.add_inequality_constraint(  # type: ignore
                lambda x, g: _linewidth_cons(x, g, settings, optimization),  # type: ignore
                optimization.nlopt_line_width_tol,
            )

        if optimization.apply_connectivity:
            solver.add_inequality_constraint(  # type: ignore
                lambda x, g: _connectivity_cons(x, g, settings, optimization),  # type: ignore
                optimization.connectivity_tolerance,
            )

        solver.set_param("dual_ftol_rel", 1e-8)  # type: ignore

        # Note: this needs to be set _after_ you set use_epsavg above, or that
        # doesn't work, and then things get bad.
        solver.set_min_objective(  # type: ignore
            lambda x, g: settings.nlopt_objective_f(x, g, optimization)  # type: ignore
        )

        weights[:] = solver.optimize(weights)  # type: ignore

        save_output(weights, settings, optimization, sigmoid_bias, history_fpath)
        optimization.last_completed_index = i

    optimal_design_weights = filter_and_project(
        weights[:],
        settings,
        optimization,
    )

    save_output(weights, settings, optimization, 0, history_fpath, binarize=True)

    return optimal_design_weights
