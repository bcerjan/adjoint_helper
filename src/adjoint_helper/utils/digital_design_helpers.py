import logging

import numpy as np

logger = logging.getLogger(__name__)


def digital_circle(
    x_size: int, y_size: int, rad: int
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Returns an array with ones everywhere except for a circular region
    in the middle where it is zero. Radius is specified in grid units."""

    if 2 * rad > x_size or 2 * rad > y_size:
        raise ValueError("Radius is bigger than the unit size!")

    if x_size % 2 != 1:
        logger.warning("X size is even so the center will be off by one pixel")

    if y_size % 2 != 1:
        logger.warning("Y size is even so the center will be off by one pixel")

    out = np.ones((x_size, y_size))

    x_cent = x_size // 2
    y_cent = y_size // 2
    y, x = np.ogrid[:x_size, :y_size]

    mask = (x - x_cent) ** 2 - (y - y_cent) ** 2 <= rad**2

    out[mask] = 0

    return out
