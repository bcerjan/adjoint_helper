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

from .base_settings import SimulationSettings as BaseSimulationSettings
from .base_settings import Edge, OptimizationSettings
from .objective_factory import get_physics_objective, PhysicsObjective


class SimulationSettings(BaseSimulationSettings):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. Import from here so that registration of
    `get_objective()` behaves correctly. Do not directly inherit from `base_settings.py`.
    """

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
        data_dir: str,
        connected_sides: list[Edge] = [],
        enforce_symmetry: bool = True,
        design_region_resolution: list[int] | int | None = None,
    ):
        super().__init__(
            wavelength=wavelength,
            resolution=resolution,
            matInd=matInd,
            subInd=subInd,
            bgInd=bgInd,
            designX=designX,
            designY=designY,
            history_fname=history_fname,
            data_dir=data_dir,
            connected_sides=connected_sides,
            enforce_symmetry=enforce_symmetry,
            design_region_resolution=design_region_resolution,
        )

    def get_objective(self, optimization: OptimizationSettings) -> PhysicsObjective:
        return get_physics_objective(self, optimization)
