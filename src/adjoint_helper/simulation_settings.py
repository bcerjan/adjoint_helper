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

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from enum import Enum
from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.mask_region import MaskRegion
from typing import Callable, Any


class Edge(Enum):
    BOTTOM = 0
    LEFT = 1
    TOP = 2
    RIGHT = 3


Optimization_Func = Callable[
    [npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
]


class SimulationSettings(ABC):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms
    """

    nx_design: int
    ny_design: int
    num_wavelengths: int
    wavelength: float
    resolution: int
    optResolution: int
    matInd: float
    subInd: float
    bgInd: float
    designX: float
    designY: float
    baseline_optimization_value: float
    needs_baseline: bool
    enforce_symmetry: bool
    history_fname: str
    data_dir: str
    connected_sides: list[int]

    def __init__(
        self,
        wavelength: float,
        resolution: int,
        matInd: float,
        subInd: float,
        bgInd: float,
        designX: float,
        designY: float,
        history_fname: str,
        data_dir: str,
        connected_sides: list[int] = [],
        enforce_symmetry: bool = True,
    ):
        self.obj = []
        self.data = []
        self.weights = []

        self.sigmoid_bias = 8
        self.sigmoid_threshold = 0.5

        self.wavelength = wavelength
        self.num_wavelengths = 1
        self.resolution = resolution
        self.optResolution = 2 * resolution  # max(2 * resolution, 200)
        self.matInd = matInd
        self.subInd = subInd
        self.bgInd = bgInd
        self.designX = designX
        self.designY = designY

        self.nx_design = round(designX * self.optResolution) + 1
        self.ny_design = round(designY * self.optResolution) + 1

        self.enforce_symmetry = enforce_symmetry
        self.history_fname = history_fname
        self.data_dir = data_dir
        self.connected_sides = connected_sides
        self.baseline_optimization_value = -np.inf
        self.needs_baseline = True

    def total_n(self) -> int:
        return self.nx_design * self.ny_design

    @abstractmethod
    def create_geometry(self) -> list[Any]:  # possibly make a generic?
        pass

    @abstractmethod
    def create_opt(
        self, optimization: OptimizationSettings
    ) -> Any:  # possibly make a generic?
        pass

    def apply_symmetry(
        self, weights: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return weights

    def border_masks(self, optimization: OptimizationSettings) -> list[MaskRegion]:
        return []
