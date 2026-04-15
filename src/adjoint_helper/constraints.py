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
import meep.adjoint as mpa  # type: ignore
from autograd import numpy as npa, tensor_jacobian_product, grad  # type: ignore
from typing import Tuple
from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.optimization_settings import OptimizationSettings


def filter_and_project(
    weights: np.ndarray[(int), np.dtype[np.float_]],
    settings: SimulationSettings,
    optimization: OptimizationSettings,
) -> np.ndarray[(int), np.dtype[np.float_]]:
    """A differentiable function to filter and project the design weights.

    Args:
      weights: design weights as a flattened (1D) array.
      settings: SimulationSettings containing the geometrical parameters
      optimization: OptimizationSettings containing optimization parameters

    Returns:
      The mapped design weights as a 1D array.
    """
    weights = weights.reshape(settings.nx_design, settings.ny_design)
    masks = settings.border_masks(optimization)

    for mask in masks:
        weights = npa.where(mask.locations, mask.value, weights)  # type: ignore

    if settings.enforce_symmetry:
        weights = settings.apply_symmetry(weights)  # type: ignore

    weights_filtered = mpa.conic_filter(  # type: ignore
        weights,
        optimization.filter_radius,
        settings.designX,
        settings.designY,
        [settings.optResolution],
    )

    if optimization.use_epsavg:
        if optimization.use_smoothed_projection:
            weights_projected = mpa.smoothed_projection(  # type: ignore
                weights_filtered,
                optimization.sigmoid_bias,
                optimization.sigmoid_threshold,
                settings.optResolution,
            )
        else:
            return weights_filtered.flatten()  # type: ignore
    else:
        weights_projected = mpa.tanh_projection(  # type: ignore
            weights_filtered,
            optimization.sigmoid_bias,
            optimization.sigmoid_threshold,
        )

    return weights_projected.flatten()  # type: ignore


def line_width_and_spacing_constraint(
    weights: np.ndarray[(int), np.dtype[np.float_]],
    gradient: np.ndarray[(int), np.dtype[np.float_]],
    settings: SimulationSettings,
    optimization: OptimizationSettings,
) -> Tuple[float, np.ndarray[(int), np.dtype[np.float_]]]:
    """Constraint function for the minimum line width and spacing.

    Args:
      weights: 1D array containing design weights
      gradient: the Jacobian matrix, modified in place.
      settings: SimulationSettings containing the geometrical parameters
      optimization: OptimizationSettings containing optimization parameters

    Returns:
      The value of the constraint function (a scalar) and the gradient.
    """

    # Possibly all need to be added to OptimizationSettings
    # Probably in a sub-class or something.
    a1 = 1e-3  # hyper parameter (primary)
    b1 = 0  # hyper parameter (secondary)
    # hyper parameter (constant factor and exponent)
    c0 = 1e6 * (optimization.filter_radius * 1 / settings.resolution) ** 4
    # gradient[:, 0] = -a1

    def filter_func(
        a: np.ndarray[(int), np.dtype[np.float_]],
    ) -> np.ndarray[(int), np.dtype[np.float_]]:
        return mpa.conic_filter(  # type: ignore
            a.reshape(settings.nx_design, settings.ny_design),
            optimization.filter_radius,
            settings.designX,
            settings.designY,
            [settings.optResolution],
        )

    def threshold_func(
        a: np.ndarray[(int), np.dtype[np.float_]],
    ) -> np.ndarray[(int), np.dtype[np.float_]]:
        return mpa.tanh_projection(  # type: ignore
            a, optimization.sigmoid_bias, optimization.sigmoid_threshold
        )

    def M1(a: np.ndarray[(int), np.dtype[np.float_]]) -> float:
        return mpa.constraint_solid(  # type: ignore
            a, c0, optimization.sigmoid_erosion, filter_func, threshold_func, 1
        )

    def M2(a: np.ndarray[(int), np.dtype[np.float_]]) -> float:
        return mpa.constraint_void(  # type: ignore
            a, c0, optimization.sigmoid_dilation, filter_func, threshold_func, 1
        )

    g1 = grad(M1)(weights)  # type: ignore
    g2 = grad(M2)(weights)  # type: ignore

    gradient[:] = g1.flatten() + g2.flatten()  # type: ignore

    t1 = (M1(weights) - b1) / a1
    t2 = (M2(weights) - b1) / a1

    return npa.max([t1, t2]), gradient  # type: ignore


def connectivity_constraint(
    weights: np.ndarray[(int), np.dtype[np.float_]],
    # gradient: np.ndarray,
    settings: SimulationSettings,
    optimization: OptimizationSettings,
) -> Tuple[float, np.ndarray[(int), np.dtype[np.float_]]]:
    """Applies connectivity constraint

    Args:
      weights: Weights for the design variables
      gradient: Gradient of the design variables
      settings: SimulationSettings containing the geometrical parameters
      optimization: OptimizationSettings containing optimization parameters

    Returns:
      Result from connectivity -- negative indicates connected as well as the gradient
    """
    if len(settings.connected_sides) == 0:
        return (0, weights)
    weights = weights[:]
    proj = filter_and_project(weights, settings, optimization).reshape(
        settings.nx_design, settings.ny_design
    )  # filter and project flattens, so we need to reshape here

    aggT = np.ones_like(proj) * np.inf
    aggGrad = np.zeros_like(proj)

    # These are the hyperparameters for our connectivity constraint. Maybe
    # should be moved into OptimizationSettings
    p = 3.0
    cond_s = 1e4
    thresh = 50

    # Need to store f0 and dJ_du per-rotation, then return the max value of f0 and
    # map the dJ_du's based on if they are connected or not yet.
    # Not sure exactly how to do that -- if positive f0(/T?) for all rotations, pick
    # largest dJ_du for that location
    # for i in range(0,4):
    for i in settings.connected_sides:
        rot = npa.rot90(proj, i).flatten()  # type: ignore
        T, _, dJ_du = mpa.constraint_connectivity(  # type: ignore
            rot,  # type: ignore
            settings.nx_design,
            1,
            settings.ny_design,
            p=p,
            cond_s=cond_s,
            thresh=thresh,
        )

        # Need to un-rotate to preserve orientation
        # T = npa.rot90(T.reshape(settings.nx_design, settings.ny_design), -i)
        # dJ_du = npa.rot90(dJ_du.reshape(settings.nx_design, settings.ny_design), -i)
        T = T.reshape(settings.nx_design, settings.ny_design)  # type: ignore
        dJ_du = dJ_du.reshape(settings.nx_design, settings.ny_design)  # type: ignore

        cond = T < aggT  # type: ignore
        aggT = npa.where(cond, T, aggT)  # type: ignore
        aggGrad = npa.where(cond, dJ_du, aggGrad)  # type: ignore

    # This is copied from the mpa.constraint_connectivity code as we need the
    # FOM for our assembled temperature, not raw temp from any rotation or
    # combination of rotations
    def heat_func(x: np.ndarray[(int), np.dtype[np.float_]]) -> float:
        return npa.sum(x**p) ** (1 / p) / thresh  # type: ignore

    f0 = heat_func(aggT) - 1  # type: ignore

    aggGrad = np.squeeze(aggGrad)  # type: ignore

    temp = tensor_jacobian_product(filter_and_project, 0)(  # type: ignore
        weights, settings, optimization, aggGrad.flatten()
    )

    # gradient[:] = temp.flatten()
    optimization.connectivity.append(f0)  # type: ignore
    return f0, temp.flatten()  # type: ignore
