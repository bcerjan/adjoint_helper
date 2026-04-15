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
    connectivity, grad[:] = connectivity_constraint(weights, settings, optimization)

    return connectivity


def run_nlopt_optimization(
    settings: SimulationSettings, optimization: OptimizationSettings
) -> np.ndarray[(int), np.dtype[np.float_]]:
    lb = np.zeros((settings.total_n(),))
    ub = np.ones((settings.total_n(),))
    weights = np.ones(settings.total_n()) * 0.5

    for mask in settings.border_masks(optimization):
        weights[mask.locations.flatten()] = mask.value

    history_fpath = settings.data_dir + settings.history_fname

    optimization.use_damping = True

    for i, sigmoid_bias in enumerate(optimization.sigmoid_biases):
        optimization.sigmoid_bias = sigmoid_bias
        solver = nlopt.opt(nlopt.LD_CCSAQ, settings.total_n())  # type: ignore
        solver.set_lower_bounds(lb)  # type: ignore
        solver.set_upper_bounds(ub)  # type: ignore

        solver.set_maxeval(optimization.max_evals[i])  # type: ignore

        if sigmoid_bias >= optimization.sigmoid_bias_threshold:
            optimization.use_epsavg = True
            # eps_avg and damping are (somewhat) mutually exclusive, see:
            # https://github.com/NanoComp/photonics-opt-testbed/issues/31#issuecomment-1370041394
            optimization.use_damping = False
        else:
            optimization.use_epsavg = False
            optimization.use_damping = True

        if sigmoid_bias > optimization.line_width_sigmoid_threshold:
            optimization.apply_linewidth = True
            solver.add_inequality_constraint(  # type: ignore
                lambda x, g: _linewidth_cons(x, g, settings, optimization),  # type: ignore
                optimization.line_width_tol,
            )

        if (
            sigmoid_bias > optimization.connectivity_sigmoid_threshold
            and optimization.do_connectivity
        ):
            optimization.apply_connectivity = True
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

    optimal_design_weights = filter_and_project(
        weights[:],
        settings,
        optimization,
    )

    save_output(weights, settings, optimization, 0, history_fpath, binarize=True)

    return optimal_design_weights
