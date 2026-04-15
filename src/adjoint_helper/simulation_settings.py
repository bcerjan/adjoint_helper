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
import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore
from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.mask_region import MaskRegion


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

    def total_n(self) -> int:
        return self.nx_design * self.ny_design

    @abstractmethod
    def create_geometry(self) -> list[mp.GeometricObject]:
        pass

    @abstractmethod
    def create_opt(self, optimization: OptimizationSettings) -> mpa.OptimizationProblem:
        pass

    @abstractmethod
    def apply_symmetry(self, weights: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        pass

    @abstractmethod
    def border_masks(self, optimization: OptimizationSettings) -> list[MaskRegion]:
        pass

    # Default optimization function for nlopt calling.
    # Can be customized if your mpa objective function has more returns
    # For simple cases (single freq. single mpa objective), this should be sufficient
    # Needs to be a minimization objective
    def nlopt_objective_f(
        self,
        weights: np.ndarray[(int), np.dtype[np.float_]],
        grad: np.ndarray[(int), np.dtype[np.float_]],
        optimization: OptimizationSettings,
    ) -> float:
        from adjoint_helper.constraints import filter_and_project
        from autograd import tensor_jacobian_product  # type: ignore

        opt = self.create_opt(optimization)

        f0, dJ_du = opt([filter_and_project(weights, self, optimization)])  # type: ignore

        grad[:] = tensor_jacobian_product(filter_and_project, 0)(
            weights,
            self,
            optimization,
            dJ_du,
        )

        optimization.obj.append(np.real(f0))  # type: ignore
        optimization.weights.append(weights.copy())
        optimization.data.append(f0)  # type: ignore

        print(f"Iteration: {len(optimization.weights)}, objective: {f0[0]:.4e}\n")
        print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}\n")

        return f0[0]  # type: ignore
