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
from .defs import PhysicsObjective, MaskRegion, WeightsType
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Union, TypeVar, Generic


W = TypeVar("W", bound="WeightsType")
S = TypeVar("S", bound="SimulationSettingsBase")


class OptimizationSettings(ABC):
    """
    Class to store necessary optimization parameters as well as optimization
    history.
    """

    obj: list[np.float64]
    data: list[np.float64]
    weights: list[npt.NDArray[np.float64]]
    grad: list[np.float64]
    connectivity: list[float]
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

    @abstractmethod
    def optimize(self, settings: SimulationSettingsBase) -> WeightsType:
        pass


class SimulationSettingsBase(ABC):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. This version of the class is intended for internal
    use only. If you are importing from here, stop, and import from export_settings.py
    instead.
    """

    n_design_regions: int
    wavelength: float
    resolution: int
    matInd: float
    subInd: float
    bgInd: float
    baseline_optimization_value: float
    needs_baseline: bool
    enforce_symmetry: bool
    history_fname: str
    data_dir: Path

    def __init__(
        self,
        wavelength: float,
        resolution: int,
        matInd: float,
        subInd: float,
        bgInd: float,
        history_fname: str,
        data_dir: str | Path,
        enforce_symmetry: bool = True,
        n_design_regions: int = 1,
    ):

        self.wavelength = wavelength
        self.resolution = resolution

        self.matInd = matInd
        self.subInd = subInd
        self.bgInd = bgInd

        self.n_design_regions = n_design_regions

        self.enforce_symmetry = enforce_symmetry
        self.history_fname = history_fname
        self.data_dir = Path(data_dir).resolve()

        self.baseline_optimization_value = -np.inf
        self.needs_baseline = True

    @abstractmethod
    def total_n(self) -> Union[list[int], int]:
        pass

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

    def border_masks(
        self, filter_radius: float
    ) -> list[MaskRegion] | MaskRegion | None:
        return None


class SimulationSettings(SimulationSettingsBase, ABC, Generic[W]):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. This version of the class is intended for internal
    use only. If you are importing from here, stop, and import from export_settings.py
    instead.
    """

    @abstractmethod
    def filter_and_project(self, weights: W, optimization: OptimizationSettings) -> W:
        pass

    @abstractmethod
    def line_width_and_spacing(
        self, weights: W, optimization: OptimizationSettings
    ) -> tuple[float, W] | list[tuple[float, npt.NDArray[np.float64]]]:
        pass

    @abstractmethod
    def connectivity_constraint(
        self, weights: W, optimization: OptimizationSettings
    ) -> tuple[float, W] | list[tuple[float, npt.NDArray[np.float64]]]:
        pass

    @abstractmethod
    def apply_symmetry(self, weights: W) -> W:
        return weights
