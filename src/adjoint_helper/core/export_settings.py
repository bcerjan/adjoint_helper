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

__all__ = ["SingleRegionSettings", "MultiRegionSettings"]

from numpy import float64
import numpy as np
from numpy._typing import NDArray
from typing import TypeVar, Any
import hashlib
import json

from .base_settings import (
    SimulationSettingsBase,
    OptimizationSettings,
    get_physics_objective,
)
from .defs import (
    Edge,
    PhysicsObjective,
    MaskRegion,
    WeightsType,
    ConstraintReturnType,
    RawWeightsType,
)

from .constraints import (
    filter_and_project_single,
    line_width_and_spacing_constraint_single,
    connectivity_constraint_single,
)

from pydantic import model_validator, computed_field

W = TypeVar("W", bound="WeightsType")
S = TypeVar("S", bound="SimulationSettingsBase")


class SingleRegionSettings(SimulationSettingsBase):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. Import from here so that registration of
    `get_objective()` behaves correctly. For single design region simulations.
    """

    # nx_design: int = 0
    # ny_design: int = 0
    design_region_resolution: int = 0
    designX: float
    designY: float
    connected_sides: list[Edge] = []

    # def __init__(
    #     self,
    #     resolution: int,
    #     designX: float,
    #     designY: float,
    #     history_fname: str,
    #     data_dir: str,
    #     connected_sides: list[Edge] = [],
    #     enforce_symmetry: bool = True,
    #     design_region_resolution: int | None = None,
    # ):

    #     super().__init__(
    #         resolution=resolution,
    #         history_fname=history_fname,
    #         data_dir=data_dir,
    #         enforce_symmetry=enforce_symmetry,
    #         n_design_regions=1,
    #     )

    #     self.designX = designX
    #     self.designY = designY
    #     if design_region_resolution is None:
    #         self.design_region_resolution = max(2 * resolution, 200)
    #     else:
    #         self.design_region_resolution = design_region_resolution

    #     self.nx_design = round(designX * self.design_region_resolution) + 1
    #     self.ny_design = round(designY * self.design_region_resolution) + 1
    #     self.connected_sides = connected_sides

    @model_validator(mode="before")
    @classmethod
    def fix_design_region_res(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d_res = data.get("design_region_resolution")  # type: ignore
            res = data.get["resolution"]  # type: ignore

            if isinstance(res, int):
                if (isinstance(d_res, int) and d_res <= 0) or d_res is None:
                    data["design_region_resolution"] = max(2 * res, 200)

        return data  # type: ignore

    @computed_field
    @property
    def nx_design(self) -> int:
        return int(round(self.designX * self.design_region_resolution)) + 1

    @computed_field
    @property
    def ny_design(self) -> int:
        return int(round(self.designY * self.design_region_resolution)) + 1

    def get_objective(
        self,
        optimization: OptimizationSettings,
    ) -> PhysicsObjective:
        return get_physics_objective(self, optimization)

    def is_multi_region(self) -> bool:
        return False

    def total_n(self) -> int:
        return self.nx_design * self.ny_design

    def total_n_raw(self) -> int:
        return self.total_n()

    def apply_symmetry(self, weights: NDArray[float64]) -> WeightsType:
        return weights

    def get_masks(self, filter_radius: float) -> list[MaskRegion] | MaskRegion | None:
        return None

    def filter_and_project(
        self,
        weights: WeightsType,
        optimization: OptimizationSettings,
    ) -> RawWeightsType:

        return filter_and_project_single(
            weights=weights if not isinstance(weights, list) else weights[0],
            design_region_resolution=self.design_region_resolution,
            designX=self.designX,
            designY=self.designY,
            optimization=optimization,
        )

    def line_width_and_spacing(
        self,
        weights: WeightsType,
        optimization: OptimizationSettings,
    ) -> ConstraintReturnType:
        return line_width_and_spacing_constraint_single(
            weights=weights if not isinstance(weights, list) else weights[0],
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
        weights: WeightsType,
        optimization: OptimizationSettings,
    ) -> ConstraintReturnType:
        return connectivity_constraint_single(
            weights=weights if not isinstance(weights, list) else weights[0],
            connected_sides=self.connected_sides,
            design_region_resolution=self.design_region_resolution,
            designX=self.designX,
            designY=self.designY,
            optimization=optimization,
        )

    def weightslike_to_raw(self, obj: WeightsType) -> RawWeightsType:
        return np.ravel(obj[0]) if isinstance(obj, list) else np.ravel(obj)

    def raw_to_weightslike(self, obj: RawWeightsType) -> WeightsType:
        """Reshapes to 2D for plotting easier interpretation"""
        return obj.reshape(self.nx_design, self.ny_design)

    def get_fingerprint(self) -> str:
        core_params: dict[str, Any] = {
            "resolution": self.resolution,
            "total_n": self.total_n_raw(),
            "n_design_regions": self.n_design_regions,
            "designX": self.designX,
            "designY": self.designY,
            "design_region_resolution": self.design_region_resolution,
        }

        param_str = json.dumps(core_params, sort_keys=True)
        return hashlib.sha3_256(param_str.encode()).hexdigest()


class MultiRegionSettings(SimulationSettingsBase):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. Import from here so that registration of
    `get_objective()` behaves correctly. For single design region simulations.
    """

    design_region_resolution: list[int] = []
    designX: list[float]
    designY: list[float]
    connected_sides: list[list[Edge]] = []

    # def __init__(
    #     self,
    #     resolution: int,
    #     designX: list[float],
    #     designY: list[float],
    #     history_fname: str,
    #     data_dir: str,
    #     connected_sides: list[list[Edge]] = [],
    #     enforce_symmetry: bool = True,
    #     design_region_resolution: int | list[int] | None = None,
    # ):

    #     super().__init__(
    #         resolution=resolution,
    #         history_fname=history_fname,
    #         data_dir=data_dir,
    #         enforce_symmetry=enforce_symmetry,
    #         n_design_regions=n_design_regions,
    #     )

    #     self.designX = designX
    #     self.designY = designY

    #     if len(designX) != len(designY):
    #         raise ValueError("Must have same number of designX and designY")

    #     if design_region_resolution is None:
    #         self.design_region_resolution = [2 * resolution for _ in designX]
    #     elif isinstance(design_region_resolution, int):
    #         self.design_region_resolution = [design_region_resolution for _ in designX]
    #     else:
    #         self.design_region_resolution = design_region_resolution

    #     if len(designX) != len(self.design_region_resolution):
    #         raise ValueError(
    #             "Must have same number of design regions as design region resolutions"
    #         )

    #     nx_design: list[int] = []
    #     ny_design: list[int] = []

    #     n_design_regions = len(designX)

    #     for i in range(n_design_regions):
    #         nx_design[i] = round(designX[i] * self.design_region_resolution[i]) + 1
    #         ny_design[i] = round(designY[i] * self.design_region_resolution[i]) + 1

    #     if len(connected_sides) != n_design_regions and len(connected_sides) != 0:
    #         raise ValueError(
    #             "If you have any connected design regions, you muse supply connections for all of them. To indicate that connectivity should not be applied to a region, insert an empty list at that index."
    #         )

    #     if len(connected_sides) == 0:
    #         self.connected_sides = [[] for _ in range(n_design_regions)]

    @model_validator(mode="before")
    @classmethod
    def fix_design_region_res(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d_res = data.get("design_region_resolution")  # type: ignore
            res = data.get["resolution"]  # type: ignore
            n = len(data.get["designX"])  # type: ignore

            if isinstance(res, int) and n > 0:
                if (isinstance(d_res, int) and d_res <= 0) or d_res is None:
                    data["design_region_resolution"] = [max(2 * res, 200)] * n

        return data  # type: ignore

    @model_validator(mode="after")
    def update_n_design_regions(self) -> "MultiRegionSettings":
        self.n_design_regions = len(self.designX)
        return self

    @model_validator(mode="after")
    def validate_lengths(self) -> "MultiRegionSettings":
        if len(self.designX) != len(self.designY):
            raise ValueError(
                f"Mismatch: designX (len {len(self.designX)}) "
                f"must match designY (len {len(self.designY)})"
            )

        if len(self.designX) != len(self.design_region_resolution):
            raise ValueError(
                f"Mismatch: designX (len {len(self.designX)}) "
                f"must match designY (len {len(self.designY)})"
            )

        return self

    @computed_field
    @property
    def nx_design(self) -> list[int]:
        nx: list[int] = []
        for i in range(self.n_design_regions):
            nx[i] = int(round(self.designX[i] * self.design_region_resolution[i])) + 1
        return nx

    @computed_field
    @property
    def ny_design(self) -> list[int]:
        ny: list[int] = []
        for i in range(self.n_design_regions):
            ny[i] = int(round(self.designY[i] * self.design_region_resolution[i])) + 1
        return ny

    def get_objective(
        self,
        optimization: OptimizationSettings,
    ) -> PhysicsObjective:
        return get_physics_objective(self, optimization)

    def total_n(self) -> list[int]:
        return [
            self.nx_design[i] * self.ny_design[i] for i in range(self.n_design_regions)
        ]

    def is_multi_region(self) -> bool:
        return True

    def total_n_raw(self) -> int:
        return np.sum(self.total_n())

    def apply_symmetry(self, weights: list[NDArray[float64]]) -> WeightsType:
        return weights

    def get_masks(self, filter_radius: float) -> list[MaskRegion]:
        return []

    def filter_and_project(
        self,
        weights: WeightsType,
        optimization: OptimizationSettings,
    ) -> RawWeightsType:

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

        return self.weightslike_to_raw(ret)

    def line_width_and_spacing(
        self,
        weights: WeightsType,
        optimization: OptimizationSettings,
    ) -> ConstraintReturnType:

        grads: list[NDArray[float64]] = []
        val = -np.inf
        # We return largest val to force the constraint.
        # gradients for other regions should be small at that point
        for i in range(self.n_design_regions):
            v, g = line_width_and_spacing_constraint_single(
                weights=weights[i],
                designX=self.designX[i],
                designY=self.designY[i],
                nx_design=self.nx_design[i],
                ny_design=self.ny_design[i],
                resolution=self.resolution,
                design_region_resolution=self.design_region_resolution[i],
                optimization=optimization,
            )

            grads.append(g)
            if v > val:
                val = v

        return (val, self.weightslike_to_raw(grads))

    def connectivity_constraint(
        self,
        weights: WeightsType,
        optimization: OptimizationSettings,
    ) -> ConstraintReturnType:
        grads: list[NDArray[float64]] = []
        val = -np.inf
        for i in range(self.n_design_regions):
            v, g = connectivity_constraint_single(
                weights=weights[i],
                connected_sides=self.connected_sides[i],
                design_region_resolution=self.design_region_resolution[i],
                designX=self.designX[i],
                designY=self.designY[i],
                optimization=optimization,
            )
            grads.append(g)
            if v > val:
                val = v

        return (val, self.weightslike_to_raw(grads))

    def weightslike_to_raw(self, obj: WeightsType) -> RawWeightsType:
        """Parses the provided list of arrays into a contiguous array. Flattens
        completely.

        Args:
            obj (list[NDArray[float64]]): Data to be flattened -- weights, gradients,...

        Returns:
            NDArray[float64]: Flattened and concatenated result
        """
        ret: list[NDArray[float64]] = []
        for i in range(self.n_design_regions):
            ret[i] = np.ravel(obj[i])

        return np.concatenate(ret)

    def raw_to_weightslike(self, obj: RawWeightsType) -> WeightsType:
        """Unflattens the provided array based on the number of grid points
        in each sub-array that composes it. Reshapes to 2D array

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
            ret.append(
                obj[start_idx:stop_idx].reshape(self.nx_design[i], self.ny_design[i])
            )
            start_idx = stop_idx + 1

        return ret

    def get_fingerprint(self) -> str:
        core_params: dict[str, Any] = {
            "resolution": self.resolution,
            "total_n": self.total_n_raw(),
            "n_design_regions": self.n_design_regions,
            "designX": self.designX,
            "designY": self.designY,
            "design_region_resolution": self.design_region_resolution,
        }

        param_str = json.dumps(core_params, sort_keys=True)
        return hashlib.sha3_256(param_str.encode()).hexdigest()
