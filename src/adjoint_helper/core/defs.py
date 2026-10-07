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
    "MaskRegion",
    "ObjectiveFunction",
    "ObjectiveReturn",
    "OptimizationFunction",
    "RawWeightsType",
    "WeightsType",
]

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Annotated,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# import numpy as np
from pydantic import BaseModel, ConfigDict

from adjoint_helper.core.differentiation_defs import anp

from .pydantic_defs import BoolArray2D, FloatArray1D, FloatArray2D


# NEEDS TESTING TO VERIFY THESE ARE CORRECT
class Edge(Enum):
    BOTTOM = 0
    LEFT = 1
    TOP = 2
    RIGHT = 3


type WeightsType = Annotated[
    anp.ndarray[tuple[int, int], anp.dtype[anp.float64 | anp.float32]], FloatArray2D
]
"""Weights laid out 'correctly' (that is, a 2D matrix).
"""

type RawWeightsType = Annotated[
    anp.ndarray[tuple[int], anp.dtype[anp.float64 | anp.float32]], FloatArray1D
]
"""Weights that have been flattened into 1D. Could be either for a single
design region or for several that have been concatenated.
"""

type ConstraintReturnType = tuple[float, RawWeightsType]
"""Return type for constraints (linewidth/connectivity)
"""


@dataclass
class Gradient:
    grad: RawWeightsType


@dataclass
class ParameterReturn:
    parameter: list[Gradient] = field(default_factory=list)


@dataclass
class ObjectiveFunctionReturn:
    objective: list[ParameterReturn] = field(default_factory=list)
    val: list[float] = field(default_factory=list)


@dataclass
class DesignRegionReturn:
    design_region: list[ObjectiveFunctionReturn] = field(default_factory=list)


SimulationReturn = tuple[list[float], list[list[list[RawWeightsType]]]]
""" Type that we expect from the simulation / backend. Order is: 
(design region, objective function, parameter, grad)"""

type ObjectiveReturn = tuple[list[float], RawWeightsType]
""" Type that we return from a physics objective -- a list of objective values and
a single 1D vector with exactly one gradient value for each design parameter. For most
optimizers, the objective values get simply summed, so if you want to weight them
differently, do that in a custom objective function. 

In general, your objective should probably return only a single value for
relatively simple simulations (as noted, if you use the defaults, multiple
objectives will simply be summed). For the epigraph formulation of nlopt,
then you should actually return a list of objective values (1 per
frequency/design parameter).
"""


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

    def __call__(self, weights: list[WeightsType]) -> SimulationReturn: ...


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
        locations (npt.NDArray[anp.bool_]): Boolean array for where the mask
            should be enforced. Must be the same size as the design region.
        value (float): What value should be enforced at these positions?
            Should be in the range [0, 1] (inclusive)
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    locations: BoolArray2D  # 1 where value should be enforced, 0 otherwise
    value: float  # value to enforce at positions, should be in [0, 1]


T = TypeVar("T", bound=anp.ndarray)


def _mean_func(data: list[T]) -> T:
    stacked = anp.stack(data, axis=0)
    return anp.mean(stacked, axis=0)


def _max_func(data: list[T]) -> T:
    stacked = anp.stack(data, axis=0)
    return anp.max(stacked, axis=0)


def _min_func(data: list[T]) -> T:
    stacked = anp.stack(data, axis=0)
    return anp.min(stacked, axis=0)


def _sum_func(data: list[T]) -> T:
    stacked = anp.stack(data, axis=0)
    return anp.sum(stacked, axis=0)


def _abs_max_func(data: list[T]) -> T:
    stacked = anp.stack(data, axis=0)
    return anp.max(anp.abs(stacked), axis=0)


def _abs_min_func(data: list[T]) -> T:
    stacked = anp.stack(data, axis=0)
    return anp.min(anp.abs(stacked), axis=0)


_OPERATION_REGISTRY: dict[str, Callable[[list[T]], T]] = {
    "mean": _mean_func,
    "max": _max_func,
    "min": _min_func,
    "abs_max": _abs_max_func,
    "abs_min": _abs_min_func,
    "sum": _sum_func,
}


def check_operation(operation: "str") -> None:
    if operation not in _OPERATION_REGISTRY:
        allowed = list(_OPERATION_REGISTRY.keys())
        raise ValueError(
            f"Invalid operation '{operation}' in `combine_gradients`. Choose from: {allowed} "
        )


def combine_objective_values(
    obj_vals: list[RawWeightsType], operation: str = "mean"
) -> list[float]:
    """Combines objective values if there are more than one design parameter."""
    check_operation(operation)

    if len(obj_vals) == 1:
        return obj_vals[0].tolist()

    out = _OPERATION_REGISTRY[operation](obj_vals)
    return out.tolist()


def combine_gradients_by_design_region(
    grad_list: list[list[list[RawWeightsType]]],
    operation: str = "mean",
) -> list[RawWeightsType]:
    """Function to apply 'simple' transforms to a set of stacked gradients. Returns a list
    separated by design region. Operations can be: 'mean', 'max', 'abs_min', 'abs_max', 'sum'

    'mean', for example, averages all of the gradients for that design region. The others
    do something similar.
    """
    check_operation(operation)

    flattened = []
    for sublist in grad_list:
        temp = _flatten_to_arrays(sublist)
        flattened.append(_OPERATION_REGISTRY[operation](temp))

    return flattened


def combine_gradients_by_objective_function(
    grad_list: list[list[list[RawWeightsType]]],
    operation: str = "mean",
) -> list[anp.ndarray[tuple[int, int], anp.dtype[anp.float64]]]:
    """Function to apply 'simple' transforms to a set of stacked gradients. Returns a list
    separated by design region and design parameter (e.g. for epigraph optimization).
    Within each design region, each objective function is collapsed -- this means that
    each design region will end up with settings.n_design_parameters gradients in a 2D
    array so that we can do grad[:,1:] = combined_grads.T

    Operations can be: 'mean', 'max', 'abs_min', 'abs_max', 'sum'

    'mean', for example, averages all of the gradients for that design region. The others
    do something similar.

    Arguments:
        `list[list[list[RawWeightsType]]]`: backpropagated gradients from simulation

    Returns:
        `list[anp.ndarray[tuple[int,int], anp.dtype[anp.float64]]`:
    """
    check_operation(operation)

    flattened = []
    for obj_fun in grad_list:
        stacked = anp.stack(obj_fun, axis=0)
        flattened.append(_OPERATION_REGISTRY[operation]([stacked]))

    return flattened


def _flatten_to_arrays(nested: list[list[RawWeightsType]]) -> list[RawWeightsType]:
    """Should only recieve the lists for a single design region"""

    def _generator(items):
        for item in items:
            if isinstance(item, anp.ndarray) and item.ndim == 1:
                yield item
            elif isinstance(item, anp.ndarray):
                raise ValueError(
                    f"Expected only 1-dimensional ndarray, got: {item.ndim}"
                )
            elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                yield from _generator(item)
            else:
                yield item

    return list(_generator(nested))  # type: ignore
