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

import numpy as np
import numpy.typing as npt
from autograd import numpy as npa, tensor_jacobian_product, grad  # type: ignore
from typing import Tuple

from .base_settings import OptimizationSettings

from .defs import Edge
from ..vendors.meep.connectivity import constraint_connectivity
from ..vendors.meep.filters import (
    conic_filter,  # type: ignore
    constraint_solid,  # type: ignore
    constraint_void,  # type: ignore
    tanh_projection,
    smoothed_projection,  # type: ignore
)


def filter_and_project_single(
    weights: npt.NDArray[np.float64],
    design_region_resolution: int,
    designX: float,
    designY: float,
    optimization: OptimizationSettings,
) -> npt.NDArray[np.float64]:
    """Mapping function for the supplied weights

    Args:
        weights (npt.NDArray[np.float64]): 2D array of weights for the design
            region at hand. Reshaped to be nx_design x ny_design. Must already
            have had symmetry and masking applied to them.
        design_region_resolution (int): Resolution of the design region
        designX (float): Physical size in X
        designY (float): Physical size in Y
        optimization (OptimizationSettings): Optimization information

    Returns:
        npt.NDArray[np.float64]: filtered weights as a 1D array
    """
    # weights = weights.reshape(nx_design, ny_design)
    # masks = settings.border_masks(optimization.filter_radius)

    # nx_design = np.size(weights, 0)
    # ny_design = np.size(weights, 1)

    # for mask in masks:
    #     weights = npa.where(mask.locations, mask.value, weights)  # type: ignore

    # if enforce_symmetry:
    #     weights = settings.apply_symmetry(weights)  # type: ignore

    weights_filtered = conic_filter(  # type: ignore
        weights,  # type: ignore
        optimization.filter_radius,
        designX,
        designY,
        [design_region_resolution],
    )

    if optimization.use_epsavg:
        if optimization.use_smoothed_projection:
            weights_projected = smoothed_projection(  # type: ignore
                weights_filtered,
                optimization.sigmoid_bias,
                optimization.sigmoid_threshold,
                design_region_resolution,
            )
        else:
            return weights_filtered.flatten()  # type: ignore
    else:
        weights_projected = tanh_projection(  # type: ignore
            weights_filtered,
            optimization.sigmoid_bias,
            optimization.sigmoid_threshold,
        )

    return weights_projected.flatten()  # type: ignore


def line_width_and_spacing_constraint_single(
    weights: npt.NDArray[np.float64],
    designX: float,
    designY: float,
    nx_design: int,
    ny_design: int,
    resolution: int,
    design_region_resolution: int,
    optimization: OptimizationSettings,
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Constraint function for the minimum line width and spacing.

    Args:
        weights (npt.NDArray[np.float64]): 1D array of weights
        designX (float): Design region physical X size
        designY (float): Design region physical Y size
        nx_design (int): Integer number of pixels in the design region X
        ny_design (int): Integer number of pixels in the design region Y
        resolution (int): Resolution of the main grid
        design_region_resolution (int): Resolution of the design region
        optimization (OptimizationSettings): Optimization information

    Returns:
        Tuple[float, npt.NDArray[np.float64]]: The value of the constraint function (a scalar) and the gradient.
    """

    # Possibly all need to be added to OptimizationSettings
    # Probably in a sub-class or something.
    a1 = 1e-3  # hyper parameter (primary)
    b1 = 0  # hyper parameter (secondary)
    # hyper parameter (constant factor and exponent)
    c0 = 1e6 * (optimization.filter_radius * 1 / resolution) ** 4

    def filter_func(
        a: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return conic_filter(  # type: ignore
            x=a.reshape(
                nx_design,
                ny_design,
            ),
            radius=optimization.filter_radius,
            Lx=designX,
            Ly=designY,
            resolution=[design_region_resolution],
        )

    def threshold_func(
        a: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return tanh_projection(  # type: ignore
            a, optimization.sigmoid_bias, optimization.sigmoid_threshold
        )

    def M1(a: npt.NDArray[np.float64]) -> float:
        return constraint_solid(  # type: ignore
            a, c0, optimization.sigmoid_erosion, filter_func, threshold_func, 1
        )

    def M2(a: npt.NDArray[np.float64]) -> float:
        return constraint_void(  # type: ignore
            a, c0, optimization.sigmoid_dilation, filter_func, threshold_func, 1
        )

    g1 = grad(M1)(weights)  # type: ignore
    g2 = grad(M2)(weights)  # type: ignore

    gradi = g1.flatten() + g2.flatten()  # type: ignore

    t1 = (M1(weights) - b1) / a1
    t2 = (M2(weights) - b1) / a1

    return npa.max([t1, t2]), gradi  # type: ignore


def connectivity_constraint_single(
    weights: npt.NDArray[np.float64],
    connected_sides: list[Edge],
    optimization: OptimizationSettings,
    design_region_resolution: int,
    designX: float,
    designY: float,
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Applies connectivity constraint. Empty list of edges signals to not apply
    the constraint.

    Args:
        weights (npt.NDArray[np.float64]): Weights for the design variables reshaped into nx_design x ny_design
        connected_sides (list[Edge]): Which sides should we connect to?
        optimization (OptimizationSettings):OptimizationSettings containing optimization parameters
        design_region_resolution (int): Resolution of the design region
        designX (float): Physical size of design region in X
        designY (float): Physical size of design region in Y

    Returns:
        Tuple[float, npt.NDArray[np.float64]]: Result from connectivity -- negative
            indicates connected as well as the gradient
    """

    if len(connected_sides) == 0:
        return (0, weights)

    nx_design = np.size(weights, 0)
    ny_design = np.size(weights, 1)

    weights = weights[:]
    proj = filter_and_project_single(
        weights=weights,
        design_region_resolution=design_region_resolution,
        designX=designX,
        designY=designY,
        optimization=optimization,
    ).reshape(
        nx_design, ny_design
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
    for i in connected_sides:
        rot = npa.rot90(proj, i.value).flatten()  # type: ignore
        T, _, dJ_du = constraint_connectivity(  # type: ignore
            rot,  # type: ignore
            nx_design,
            1,
            ny_design,
            p=p,
            cond_s=cond_s,
            thresh=thresh,
        )

        # Need to un-rotate to preserve orientation
        # T = npa.rot90(T.reshape(settings.nx_design, settings.ny_design), -i)
        # dJ_du = npa.rot90(dJ_du.reshape(settings.nx_design, settings.ny_design), -i)
        T = T.reshape(nx_design, ny_design)  # type: ignore
        dJ_du = dJ_du.reshape(nx_design, ny_design)  # type: ignore

        cond = T < aggT  # type: ignore
        aggT = npa.where(cond, T, aggT)  # type: ignore
        aggGrad = npa.where(cond, dJ_du, aggGrad)  # type: ignore

    # This is copied from the mpa.constraint_connectivity code as we need the
    # FOM for our assembled temperature, not raw temp from any rotation or
    # combination of rotations
    def heat_func(x: npt.NDArray[np.float64]) -> float:
        return npa.sum(x**p) ** (1 / p) / thresh  # type: ignore

    f0 = heat_func(aggT) - 1  # type: ignore

    aggGrad = npa.squeeze(aggGrad)  # type: ignore

    temp = tensor_jacobian_product(filter_and_project_single, 0)(  # type: ignore
        weights,
        design_region_resolution,
        designX,
        designY,
        optimization,
        aggGrad.flatten(),  # type: ignore
    )

    optimization.connectivity.append(f0)
    return f0, temp.flatten()  # type: ignore
