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

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from ..vendors.meep.filters import get_conic_radius_from_eta_e  # type: ignore
from .mask_region import MaskRegion

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable
from enum import Enum


class OptimizationSettings(ABC):
    """
    Class to store necessary optimization parameters as well as optimization
    history.
    """

    obj: list[np.float64]
    data: list[np.float64]
    weights: list[npt.NDArray[np.float64]]
    grad: list[np.float64]
    connectivity: list[int]
    do_connectivity: bool
    sigmoid_threshold: float
    sigmoid_erosion: float
    sigmoid_dilation: float
    sigmoid_biases: list[float]
    sigmoid_bias: float
    sigmoid_bias_sim: float
    sigmoid_bias_threshold: float
    epigraph_tolerance: float
    connectivity_sigmoid_threshold: float
    _apply_connectivity: bool
    do_connectivity: bool
    linewidth_sigmoid_threshold: float
    _apply_linewidth: bool
    max_evals: list[int]
    filter_radius: float
    _use_epsavg: bool = False
    _use_damping: bool = True
    maximum_runtime: float
    minimum_runtime: float
    decay_by: float
    use_smoothed_projection: bool
    last_completed_index: int

    def __init__(
        self,
        minimum_size: float = 0.05,
        sigmoid_bias_threshold: float = 32,  # Sigmoid bias at which eps_avg turns on
        sigmoid_threshold: float = 0.5,  # Eta
        sigmoid_erosion: float = 0.65,  # Eta_e
        sigmoid_biases: list[float] = [4, 8, 16, 24, 32, 40],
        connectivity_sigmoid_threshold: float = 16,
        linewidth_sigmoid_threshold: float = 24,  # Sigmoid bias at which line width constraint turns on
        max_evals: list[int] | int = 10,  # if int, all biases get same number
        maximum_runtime: float = 200,
        minimum_runtime: float = 0,
        decay_by: float = 1e-6,
        use_smoothed_projection: bool = False,
        do_connectivity: bool = False,
    ):
        """"""
        evals: list[int] = []

        if type(max_evals) is list and len(sigmoid_biases) != len(max_evals):
            raise ValueError("Mismatch between length of sigmoid_biases and max_evals")

        if type(max_evals) is int:
            evals = [max_evals for _ in sigmoid_biases]
        else:
            evals = max_evals  # type: ignore

        self.obj = []
        self.data = []
        self.weights = []
        self.grad = []
        self.connectivity = []
        self.last_completed_index = -1

        self.filter_radius = get_conic_radius_from_eta_e(  # type: ignore
            minimum_size, sigmoid_erosion
        )

        self.sigmoid_bias = sigmoid_biases[0]
        self.sigmoid_biases = sigmoid_biases
        self.sigmoid_bias_threshold = sigmoid_bias_threshold
        self.sigmoid_threshold = sigmoid_threshold
        self.sigmoid_erosion = sigmoid_erosion
        self.sigmoid_dilation = 1 - sigmoid_erosion
        self.max_evals = evals
        self.maximum_runtime = maximum_runtime
        self.minimum_runtime = minimum_runtime
        self.decay_by = decay_by
        self.apply_connectivity = False
        self.apply_linewidth = False
        self.do_connectivity = do_connectivity
        self.connectivity_sigmoid_threshold = connectivity_sigmoid_threshold
        self.linewidth_sigmoid_threshold = linewidth_sigmoid_threshold
        self.use_smoothed_projection = use_smoothed_projection

    # Change this to a bunch of getters rather than this kind of function.
    # This is doing unnecessary stuff.
    # def apply_settings(self, current_sigmoid_bias: float) -> None:
    #     """Updates the optimization object to tell it to use/not use eps_avg,
    #     damping, connectivity constraint, and linewidth constraint. Also updates
    #     the internally stored `sigmoid_bias`

    #     Args:
    #         current_sigmoid_bias (float): Current bias of the optimization
    #     """
    #     self.sigmoid_bias = current_sigmoid_bias

    #     if current_sigmoid_bias >= self.sigmoid_bias_threshold:
    #         self.use_epsavg = True
    #         # eps_avg and damping are (somewhat) mutually exclusive, see:
    #         # https://github.com/NanoComp/photonics-opt-testbed/issues/31#issuecomment-1370041394
    #         self.use_damping = False
    #     else:
    #         self.use_epsavg = False
    #         self.use_damping = True

    #     if (
    #         current_sigmoid_bias > self.connectivity_sigmoid_threshold
    #         and self.do_connectivity
    #     ):
    #         self.apply_connectivity = True
    #     else:
    #         self.apply_connectivity = False

    #     if current_sigmoid_bias > self.linewidth_sigmoid_threshold:
    #         self.apply_linewidth = True
    #     else:
    #         self.apply_linewidth = False

    @abstractmethod
    def optimize(self, settings: SimulationSettings) -> npt.NDArray[np.float64]:
        pass

    @property
    def bias(self) -> float:
        return self.sigmoid_biases[self.last_completed_index + 1]

    @property
    def apply_linewidth(self) -> bool:
        return (self.bias > self.linewidth_sigmoid_threshold) or self._apply_linewidth

    @apply_linewidth.setter
    def apply_linewidth(self, val: bool):
        self._apply_linewidth = val

    @property
    def apply_connectivity(self) -> bool:
        return (
            (self.bias > self.connectivity_sigmoid_threshold) and self.do_connectivity
        ) or self._apply_connectivity

    @apply_connectivity.setter
    def apply_connectivity(self, val: bool):
        self._apply_linewidth = val

    @property
    def use_damping(self) -> bool:
        return not self.use_epsavg or self._use_damping

    @use_damping.setter
    def use_damping(self, val: bool):
        self._use_damping = val

    @property
    def use_epsavg(self) -> bool:
        return (self.bias > self.sigmoid_bias_threshold) or self._use_epsavg

    @use_epsavg.setter
    def use_epsavg(self, val: bool):
        self._use_epsavg = val


# NEEDS TESTING TO VERIFY THESE ARE CORRECT
class Edge(Enum):
    BOTTOM = 0
    LEFT = 1
    TOP = 2
    RIGHT = 3


class SimulationSettings(ABC):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms
    """

    nx_design: list[int]
    ny_design: list[int]
    n_design_regions: int
    num_wavelengths: int
    wavelength: float
    resolution: int
    design_region_resolution: list[int]
    matInd: float
    subInd: float
    bgInd: float
    designX: list[float]
    designY: list[float]
    baseline_optimization_value: float
    needs_baseline: bool
    enforce_symmetry: bool
    history_fname: str
    data_dir: Path
    connected_sides: list[Edge]

    def __init__(
        self,
        wavelength: float,
        resolution: int,
        matInd: float,
        subInd: float,
        bgInd: float,
        designX: list[float] | float,
        designY: list[float] | float,
        history_fname: str,
        data_dir: str | Path,
        design_region_resolution: list[int] | int | None = None,
        connected_sides: list[Edge] = [],
        enforce_symmetry: bool = True,
    ):

        self.wavelength = wavelength
        self.num_wavelengths = 1
        self.resolution = resolution

        self.matInd = matInd
        self.subInd = subInd
        self.bgInd = bgInd

        if isinstance(designX, (float, int)):
            designX = [designX]
        if isinstance(designY, (float, int)):
            designY = [designY]

        if isinstance(design_region_resolution, int):
            design_region_resolution = [design_region_resolution]

        if len(designX) != len(designY):
            raise ValueError(
                "Must have same number of x and y sizes for design region(s)"
            )

        if (design_region_resolution is not None) and (
            len(designX) != len(design_region_resolution)
        ):
            raise ValueError(
                "Must have same number of design regions and design region resolutions"
            )

        self.n_design_regions = len(designX)

        self.designX = designX
        self.designY = designY

        for i in range(self.n_design_regions):
            if design_region_resolution is None:
                self.design_region_resolution[i] = 2 * resolution

            self.nx_design[i] = round(designX[i] * self.design_region_resolution[i]) + 1
            self.ny_design[i] = round(designY[i] * self.design_region_resolution[i]) + 1

        self.enforce_symmetry = enforce_symmetry
        self.history_fname = history_fname
        self.data_dir = Path(data_dir).resolve()
        self.connected_sides = connected_sides
        self.baseline_optimization_value = -np.inf
        self.needs_baseline = True

    def total_n(self) -> list[int]:
        return [
            self.nx_design[i] * self.ny_design[i] for i in range(len(self.nx_design))
        ]

    @abstractmethod
    def create_geometry(self) -> list[Any]:  # possibly make a generic?
        pass

    @abstractmethod
    def create_opt(self, optimization: OptimizationSettings) -> PhysicsObjective:
        """This is the "main" method for your optimization. You can use whatever
        simulation engine you like (meep, tidy3D, legume, COMSOL, Elmer, ... ) as
        long as it can produce gradients for your objective function (whatever
        that may be). The produced function should take in weights as the only
        argument and must return (objective, gradient) (as arrays). The weights
        will be `filter_and_project`'ed before they are sent to your function,
        so you should not do that internally.

        This

        Args:
            optimization (OptimizationSettings): Everything needed for only the
            immediate simulation / optimization step. Not the auxillary stuff
            specific to each optimization method.

        Returns:
            out (PhysicsObjective): Function that takes in weights and returns
            gradients
        """
        pass

    def apply_symmetry(
        self, weights: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return weights

    def border_masks(self, filter_radius: float) -> list[MaskRegion]:
        return []


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
        out (tuple[list[npt.NDArray[np.float64]], list[list[npt.NDArray[np.float64]]]]):
            A tuple containing:
            1. An array of objective values (even for a single objective, this should
            be an array) from this iteration.

            2. A list of lists of 2D gradients. The order is objective_functions
            -> design_regions -> frequencies
    """

    def __call__(self, weights: list[npt.NDArray[np.float64]]) -> ObjectiveReturn: ...
