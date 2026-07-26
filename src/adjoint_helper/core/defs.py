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
    "ObjectiveFunction",
    "OptimizationFunction",
    "MaskRegion",
    "WeightsType",
    "RawWeightsType",
]

from enum import Enum
from typing import (
    Annotated,
    Iterable,
    List,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np
from pydantic import BaseModel, ConfigDict

from .pydantic_defs import BoolArray2D, FloatArray1D, FloatArray2D


# NEEDS TESTING TO VERIFY THESE ARE CORRECT
class Edge(Enum):
    BOTTOM = 0
    LEFT = 1
    TOP = 2
    RIGHT = 3


type WeightsType = Annotated[
    np.ndarray[Tuple[int, int], np.dtype[np.float64]], FloatArray2D
]
"""Weights laid out 'correctly' (that is, a 2D matrix).
"""

type RawWeightsType = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float64]], FloatArray1D
]
"""Weights that have been flattened into 1D. Could be either for a single
design region or for several that have been concatenated.
"""

type ConstraintReturnType = tuple[float, RawWeightsType]
"""Return type for constraints (linewidth/connectivity)
"""

SimulationReturn = tuple[list[float], list[list[list[RawWeightsType]]]]
""" Type that we expect from the simulation / backend. """

type ObjectiveReturn = tuple[list[float], RawWeightsType]
""" Type that we return from a physics objective -- a list of objective values and
a single 1D vector with exactly one gradient value for each design parameter. For most
optimizers, the objective values get simply summed, so if you want to weight them
differently, do that in a custom objective function. """


@runtime_checkable
class OptimizationFunction(Protocol):
    """
    This is a function that will be called to convert a list of 2D weights
    into objective values and gradients. It's a bit of a doozy,
    unfortunately, because it returns a lot of information for the optimization
    process.

    Succinctly it returns: (list[objective_val], list[list[list[gradient]]])
    PhysicsObjective takes in a list of weights (per design region) and returns
    a tuple containing a list of objective values (per design region) and a list
    by objective function of lists by design region of 2D arrays of gradients.

    So it's: objective(list[2D weights per design region]) -> tuple[list[objective values], list[list[list[gradients]]]

    Args:
        weights (RawWeightsType): concatenated 1D array of weights for all design regions.

    Returns:
        out (AdjointReturn):
            A tuple containing:
            1. A list of objective values (even for a single objective, this should
            be an array) from this iteration.

            2. A list of lists of 1D gradients. The order is design regions
            -> objective functions -> frequencies/parameters -> gradient
    """

    def __call__(self, weights: List[WeightsType]) -> SimulationReturn: ...


@runtime_checkable
class ObjectiveFunction(Protocol):
    """
    This is the function that will be called to convert the return values from the
    adjoint simulation into a single concatenated list of gradients for all design
    regions.

    Args:
        weights (RawWeightsType): concatenated 1D array of weights for all design regions.

    Returns:
        out (ObjectiveReturn):
            A tuple containing:
                1. A list of objective values (even for a single objective, this should
                be an array) from this iteration.

                2. Single concatenated vector of gradients for all design weights
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

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    locations: BoolArray2D  # 1 where value should be enforced, 0 otherwise
    value: float  # value to enforce at positions, should be in [0, 1]


def _mean_func(data: list[RawWeightsType]) -> RawWeightsType:
    stacked = np.stack(data, axis=0)
    return np.mean(stacked, axis=0)


def _max_func(data: list[RawWeightsType]) -> RawWeightsType:
    stacked = np.stack(data, axis=0)
    return np.max(stacked, axis=0)


def _min_func(data: list[RawWeightsType]) -> RawWeightsType:
    stacked = np.stack(data, axis=0)
    return np.min(stacked, axis=0)


def _sum_func(data: list[RawWeightsType]) -> RawWeightsType:
    stacked = np.stack(data, axis=0)
    return np.sum(stacked, axis=0)


def _abs_max_func(data: list[RawWeightsType]) -> RawWeightsType:
    stacked = np.stack(data, axis=0)
    return np.max(np.abs(stacked), axis=0)


def _abs_min_func(data: list[RawWeightsType]) -> RawWeightsType:
    stacked = np.stack(data, axis=0)
    return np.min(np.abs(stacked), axis=0)


_OPERATION_REGISTRY = {
    "mean": _mean_func,
    "max": _max_func,
    "min": _min_func,
    "abs_max": _abs_max_func,
    "abs_min": _abs_min_func,
    "sum": _sum_func,
}


def combine_gradients(
    grad_list: list[list[list[RawWeightsType]]],
    operation: str = "mean",
) -> list[RawWeightsType]:
    """Function to apply 'simple' transforms to a set of stacked gradients. Returns a list
    separated by design region. Operations can be: 'mean', 'max', 'abs_min', 'abs_max', 'sum'

    'mean', for example, averages all of the gradients for that design region. The others
    do something similar.
    """
    if operation not in _OPERATION_REGISTRY:
        allowed = list(_OPERATION_REGISTRY.keys())
        raise ValueError(
            f"Invalid operation '{operation}' in `combine_gradients`. Choose from: {allowed} "
        )

    flattened = []
    for sublist in grad_list:
        temp = _flatten_to_arrays(sublist)
        flattened.append(_OPERATION_REGISTRY[operation](temp))

    return flattened


def _flatten_to_arrays(nested: list[list[RawWeightsType]]) -> list[RawWeightsType]:
    """Should only recieve the lists for a single design region"""

    def _generator(items):
        for item in items:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                yield item
            elif isinstance(item, np.ndarray):
                raise ValueError(
                    f"Expected only 1-dimensional ndarray, got: {item.ndim}"
                )
            elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                yield from _generator(item)
            else:
                yield item

    return list(_generator(nested))  # type: ignore
