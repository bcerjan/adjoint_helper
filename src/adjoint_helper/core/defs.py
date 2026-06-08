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

__all__ = [
    "Edge",
    "ObjectiveReturn",
    "PhysicsObjective",
    "MaskRegion",
    "WeightsType",
    "RawWeightsType",
]

import numpy as np
import numpy.typing as npt
from typing import Protocol, runtime_checkable
from pydantic import BaseModel
from enum import Enum


# NEEDS TESTING TO VERIFY THESE ARE CORRECT
class Edge(Enum):
    BOTTOM = 0
    LEFT = 1
    TOP = 2
    RIGHT = 3


ObjectiveReturn = tuple[list[float], "RawWeightsType"]


type WeightsType = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"""Weights laid out 'correctly' (that is, a 2D matrix).
"""

type RawWeightsType = np.ndarray[tuple[int], np.dtype[np.float64]]
"""Weights that have been flattened into 1D. Could be either for a single
design region or for several that have been concatenated.
"""

type ConstraintReturnType = tuple[float, RawWeightsType]
"""Return type for constraints (linewidth/connectivity)
"""


@runtime_checkable
class PhysicsObjective(Protocol):
    """
    This is a function that will be called to convert a list of weights (one per
    design region) into objective values and gradients. It's a bit of a doozy,
    unfortunately, because it returns a lot of information for the optimization
    process.

    Succinctly it returns: list[weights] -> (list[objective_val], list[list[gradient]])
    PhysicsObjective takes in a list of weights (per design region) and returns
    a tuple containing a list of objective values (per design region) and a list
    by objective function of lists by design region of 2D arrays of gradients (by
    frequency).

    So it's: ()

    Args:
        weights (list[npt.NDArray[np.float64]]): A list of weights per design region
            as 1D arrays.

        settings (SimulationSettings): Simulation settings needed for calculating
            gradients / setting up the physical domain

    Returns:
        out (ObjectiveReturn):
            A tuple containing:
            1. A list of objective values (even for a single objective, this should
            be an array) from this iteration.

            2. A list of lists of 2D gradients. The order is objective_functions
            -> design_regions -> frequencies/parameters
    """

    def __call__(self, weights: RawWeightsType) -> ObjectiveReturn: ...


class MaskRegion(BaseModel):
    """Definitions for forced values for design variables.

    Args:
        locations (npt.NDArray[np.bool_]): Boolean array for where the mask
            should be enforced. Must be the same size as the design region.
        value (float): What value should be enforced at these positions?
            Should be in the range [0, 1] (inclusive)
    """

    locations: npt.NDArray[np.bool_]  # 1 where value should be enforced, 0 otherwise
    value: float  # value to enforce at positions, should be in [0, 1]
