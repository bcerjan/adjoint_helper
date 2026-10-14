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

"""
This file contains a demonstration of a simple single-region single-objective
optimization. The settings (the only hard part) are defined separately so they
can be reused in other examples.
"""
import logging

import numpy as real_np

logger = logging.getLogger(__name__)


def find_design_unit_size(
    full_shape: tuple[int, int], target_cells: tuple[int, int] | None = None
) -> tuple[int, int]:
    return (1, 1)


def digital_circle(
    x_size: int, y_size: int, rad: int
) -> real_np.ndarray[tuple[int, int], real_np.dtype[real_np.float64]]:
    """Returns an array with ones everywhere except for a circular region
    in the middle where it is zero. Radius is specified in grid units."""

    if 2 * rad > x_size or 2 * rad > y_size:
        raise ValueError("Radius is bigger than the unit size!")

    if x_size % 2 != 1:
        logger.warning(
            "Design unit X size is even so the center will be off by one pixel"
        )

    if y_size % 2 != 1:
        logger.warning(
            "Design unit Y size is even so the center will be off by one pixel"
        )

    out = real_np.ones((x_size, y_size))

    x_cent = x_size // 2
    y_cent = y_size // 2
    y, x = real_np.ogrid[:x_size, :y_size]

    mask = (x - x_cent) ** 2 + (y - y_cent) ** 2 <= rad**2

    out[mask] = 0

    return out


def digital_rectangle(
    x_size: int, y_size: int, x_edge: int, y_edge: int | None = None
) -> real_np.ndarray[tuple[int, int], real_np.dtype[real_np.float64]]:
    """Returns an array with ones everywhere except for a rectangular region
    in the middle where it is zero. Dimensions are in grid units."""

    if y_edge is None:
        y_edge = x_edge

    if x_edge > x_size or y_edge > y_size:
        raise ValueError("Rectangle is bigger than the unit size!")

    if x_size % 2 != 1:
        logger.warning(
            "Design unit X size is even so the center will be off by one pixel"
        )

    if y_size % 2 != 1:
        logger.warning(
            "Design unit Y size is even so the center will be off by one pixel"
        )

    out = real_np.ones((x_size, y_size))

    x_cent = x_size // 2
    y_cent = y_size // 2
    x_start = x_cent - x_edge // 2
    x_stop = x_cent + x_edge // 2
    y_start = y_cent - y_edge // 2
    y_stop = y_cent + y_edge // 2

    out[x_start:x_stop, y_start:y_stop] = 0

    return out
