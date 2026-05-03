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

from numpy import float64
import numpy as np
from numpy._typing import NDArray
from typing import TypeVar

from .base_settings import SimulationSettings as BaseSimulationSettings
from .base_settings import SimulationSettingsBase, OptimizationSettings
from .defs import Edge, PhysicsObjective, MaskRegion, WeightsType
from .objective_factory import get_physics_objective
from .constraints import (
    filter_and_project_single,
    line_width_and_spacing_constraint_single,
    connectivity_constraint_single,
)

W = TypeVar("W", bound="WeightsType")
S = TypeVar("S", bound="SimulationSettingsBase")


class SingleRegionSettings(BaseSimulationSettings[NDArray[float64]]):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. Import from here so that registration of
    `get_objective()` behaves correctly. For single design region simulations.
    """

    nx_design: int
    ny_design: int
    design_region_resolution: int
    designX: float
    designY: float
    connected_sides: list[Edge]

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
        connected_sides: list[Edge] = [],
        enforce_symmetry: bool = True,
        design_region_resolution: int | None = None,
    ):

        self.designX = designX
        self.designY = designY
        if design_region_resolution is None:
            self.design_region_resolution = 2 * resolution
        else:
            self.design_region_resolution = design_region_resolution

        self.nx_design = round(designX * self.design_region_resolution) + 1
        self.ny_design = round(designY * self.design_region_resolution) + 1
        self.connected_sides = connected_sides
        super().__init__(
            wavelength=wavelength,
            resolution=resolution,
            matInd=matInd,
            subInd=subInd,
            bgInd=bgInd,
            history_fname=history_fname,
            data_dir=data_dir,
            enforce_symmetry=enforce_symmetry,
            n_design_regions=1,
        )

    def get_objective(
        self,
        optimization: OptimizationSettings,
    ) -> PhysicsObjective:
        return get_physics_objective(self, optimization)

    def total_n(self) -> int:
        return self.nx_design * self.ny_design

    def apply_symmetry(self, weights: NDArray[float64]) -> NDArray[float64]:
        return weights

    def border_masks(self, filter_radius: float) -> MaskRegion | None:
        return None

    def filter_and_project(
        self,
        weights: NDArray[float64],
        optimization: OptimizationSettings,
    ) -> NDArray[float64]:

        return filter_and_project_single(
            weights=weights,
            design_region_resolution=self.design_region_resolution,
            designX=self.designX,
            designY=self.designY,
            optimization=optimization,
        )

    def line_width_and_spacing(
        self,
        weights: NDArray[float64],
        optimization: OptimizationSettings,
    ) -> tuple[float, NDArray[float64]]:
        return line_width_and_spacing_constraint_single(
            weights=weights,
            designX=self.designX,
            designY=self.designY,
            nx_design=self.nx_design,
            ny_design=self.ny_design,
            resolution=self.resolution,
            design_region_resolution=self.design_region_resolution,
            optimization=optimization,
        )

    def connectivity_constraint(
        self,
        weights: NDArray[float64],
        optimization: OptimizationSettings,
    ) -> tuple[float, NDArray[float64]]:
        return connectivity_constraint_single(
            weights=weights,
            connected_sides=self.connected_sides,
            design_region_resolution=self.design_region_resolution,
            designX=self.designX,
            designY=self.designY,
            optimization=optimization,
        )


class MultiRegionSettings(BaseSimulationSettings[list[NDArray[float64]]]):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. Import from here so that registration of
    `get_objective()` behaves correctly. For single design region simulations.
    """

    nx_design: list[int]
    ny_design: list[int]
    design_region_resolution: list[int]
    designX: list[float]
    designY: list[float]
    connected_sides: list[list[Edge]]

    def __init__(
        self,
        wavelength: float,
        resolution: int,
        matInd: float,
        subInd: float,
        bgInd: float,
        designX: list[float],
        designY: list[float],
        history_fname: str,
        data_dir: str,
        connected_sides: list[list[Edge]] = [],
        enforce_symmetry: bool = True,
        design_region_resolution: int | list[int] | None = None,
    ):

        self.designX = designX
        self.designY = designY

        if len(designX) != len(designY):
            raise ValueError("Must have same number of designX and designY")

        if design_region_resolution is None:
            self.design_region_resolution = [2 * resolution for _ in designX]
        elif isinstance(design_region_resolution, int):
            self.design_region_resolution = [design_region_resolution for _ in designX]
        else:
            self.design_region_resolution = design_region_resolution

        if len(designX) != len(self.design_region_resolution):
            raise ValueError(
                "Must have same number of design regions as design region resolutions"
            )

        nx_design: list[int] = []
        ny_design: list[int] = []

        n_design_regions = len(designX)

        for i in range(n_design_regions):
            nx_design[i] = round(designX[i] * self.design_region_resolution[i]) + 1
            ny_design[i] = round(designY[i] * self.design_region_resolution[i]) + 1

        if len(connected_sides) != n_design_regions and len(connected_sides) != 0:
            raise ValueError(
                "If you have any connected design regions, you muse supply connections for all of them. To indicate that connectivity should not be applied to a region, insert an empty list at that index."
            )

        if len(connected_sides) == 0:
            self.connected_sides = [[] for _ in range(n_design_regions)]

        super().__init__(
            wavelength=wavelength,
            resolution=resolution,
            matInd=matInd,
            subInd=subInd,
            bgInd=bgInd,
            history_fname=history_fname,
            data_dir=data_dir,
            enforce_symmetry=enforce_symmetry,
            n_design_regions=n_design_regions,
        )

    def get_objective(
        self,
        optimization: OptimizationSettings,
    ) -> PhysicsObjective:
        return get_physics_objective(self, optimization)

    def total_n(self) -> list[int]:
        return [
            self.nx_design[i] * self.ny_design[i] for i in range(self.n_design_regions)
        ]

    def apply_symmetry(self, weights: list[NDArray[float64]]) -> list[NDArray[float64]]:
        return weights

    def border_masks(self, filter_radius: float) -> list[MaskRegion]:
        return []

    def filter_and_project(
        self,
        weights: list[NDArray[float64]],
        optimization: OptimizationSettings,
    ) -> list[NDArray[float64]]:

        ret: list[NDArray[float64]] = []
        for i in range(self.n_design_regions):
            ret.append(
                filter_and_project_single(
                    weights=weights[i],
                    design_region_resolution=self.design_region_resolution[i],
                    designX=self.designX[i],
                    designY=self.designY[i],
                    optimization=optimization,
                )
            )

        return ret

    def to_flat(self, obj: list[NDArray[float64]]) -> NDArray[float64]:
        """Parses the provided list of arrays into a contiguous array. Flattens
        completely.

        Args:
            obj (list[NDArray[float64]]): Data to be flattened -- weights, gradients,...

        Returns:
            NDArray[float64]: Flattened and concatenated result
        """

        return np.concatenate(obj)

    def from_flat(self, obj: NDArray[float64]) -> list[NDArray[float64]]:
        """Unflattens the provided array based on the number of grid points
        in each sub-array that composes it.

        Args:
            obj (NDArray[float64]): Data to be unflattened -- weights, gradients,...

        Returns:
            out (list[NDArray[float64]]): The "standard" list of arrays sorted by
            design region
        """
        ret: list[NDArray[float64]] = []

        start_idx = 0
        stop_idx = 0
        totals = self.total_n()
        for i in range(self.n_design_regions):
            stop_idx = start_idx + totals[i]
            ret.append(obj[start_idx:stop_idx])
            start_idx = stop_idx + 1

        return ret

    def line_width_and_spacing(
        self,
        weights: list[NDArray[float64]],
        optimization: OptimizationSettings,
    ) -> list[tuple[float, NDArray[float64]]]:
        ret: list[tuple[float, NDArray[float64]]] = []
        for i in range(self.n_design_regions):
            ret.append(
                line_width_and_spacing_constraint_single(
                    weights=weights[i],
                    designX=self.designX[i],
                    designY=self.designY[i],
                    nx_design=self.nx_design[i],
                    ny_design=self.ny_design[i],
                    resolution=self.resolution,
                    design_region_resolution=self.design_region_resolution[i],
                    optimization=optimization,
                )
            )

        return ret

    def connectivity_constraint(
        self,
        weights: list[NDArray[float64]],
        optimization: OptimizationSettings,
    ) -> list[tuple[float, NDArray[float64]]]:
        ret: list[tuple[float, NDArray[float64]]] = []
        for i in range(self.n_design_regions):
            ret.append(
                connectivity_constraint_single(
                    weights=weights[i],
                    connected_sides=self.connected_sides[i],
                    design_region_resolution=self.design_region_resolution[i],
                    designX=self.designX[i],
                    designY=self.designY[i],
                    optimization=optimization,
                )
            )

        return ret
