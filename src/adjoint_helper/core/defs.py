import numpy as np
import numpy.typing as npt
from typing import Protocol, runtime_checkable

ObjectiveReturn = tuple[
    list[npt.NDArray[np.float64]], list[list[npt.NDArray[np.float64]]]
]


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

    def __call__(self, weights: list[npt.NDArray[np.float64]]) -> ObjectiveReturn: ...
