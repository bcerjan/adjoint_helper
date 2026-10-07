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

from adjoint_helper import anp

logger = logging.getLogger(__name__)


def digital_circle(
    x_size: int, y_size: int, rad: int
) -> anp.ndarray[tuple[int, int], anp.dtype[anp.float64]]:
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

    out = anp.ones((x_size, y_size))

    x_cent = x_size // 2
    y_cent = y_size // 2
    y, x = anp.ogrid[:x_size, :y_size]

    mask = (x - x_cent) ** 2 + (y - y_cent) ** 2 <= rad**2

    out[mask] = 0

    return out
