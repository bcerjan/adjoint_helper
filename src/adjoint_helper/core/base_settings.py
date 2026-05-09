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

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from ..vendors.meep.filters import get_conic_radius_from_eta_e  # type: ignore
from .defs import (
    PhysicsObjective,
    MaskRegion,
    WeightsType,
    RawWeightsType,
    ConstraintReturnType,
)

from pathlib import Path
from abc import ABC, abstractmethod
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
    computed_field,
)
from typing import Any, Union, TypeVar, Callable
import base64
import io


S = TypeVar("S", bound="SimulationSettingsBase")


class OptimizationSettings(BaseModel, ABC):
    """
    Class to store necessary optimization parameters as well as optimization
    history.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    obj: list[float] = []
    data: list[float] = []
    weights: list[npt.NDArray[np.float64]] = Field(default_factory=list)  # type: ignore
    grad: list[np.float64] = []
    connectivity: list[float] = []
    sigmoid_threshold: float = 0.5
    sigmoid_erosion: float = 0.65
    sigmoid_biases: list[float] = [4, 8, 16, 24, 32, 40]
    sigmoid_bias: float = -1
    sigmoid_bias_threshold: float = 32
    connectivity_sigmoid_threshold: float = 16
    _apply_connectivity: bool = False
    do_connectivity: bool = False
    linewidth_sigmoid_threshold: float = 24
    _apply_linewidth: bool = True
    max_evals: list[int] = []
    _use_epsavg: bool = False
    _use_damping: bool = True
    maximum_runtime: float = 200
    minimum_runtime: float = 0
    decay_by: float = 1e-6
    use_smoothed_projection: bool = False
    last_completed_index: int = -1
    minimum_feature_size: float = 0.05

    # def __init__(
    #     self,
    #     minimum_size: float = 0.05,
    #     sigmoid_bias_threshold: float = 32,  # Sigmoid bias at which eps_avg turns on
    #     sigmoid_threshold: float = 0.5,  # Eta
    #     sigmoid_erosion: float = 0.65,  # Eta_e
    #     sigmoid_biases: list[float] = [4.0, 8, 16, 24, 32, 40],
    #     connectivity_sigmoid_threshold: float = 16,
    #     linewidth_sigmoid_threshold: float = 24,  # Sigmoid bias at which line width constraint turns on
    #     max_evals: list[int] | int = 10,  # if int, all biases get same number
    #     maximum_runtime: float = 200,
    #     minimum_runtime: float = 0,
    #     decay_by: float = 1e-6,
    #     use_smoothed_projection: bool = False,
    #     do_connectivity: bool = False,
    # ):
    # super().__init__()
    # evals: list[int] = []

    # if type(max_evals) is list and len(sigmoid_biases) != len(max_evals):
    #     raise ValueError("Mismatch between length of sigmoid_biases and max_evals")

    # if type(max_evals) is int:
    #     evals = [max_evals for _ in sigmoid_biases]
    # else:
    #     evals = max_evals  # type: ignore

    # self.obj = []
    # self.data = []
    # self.weights = []
    # self.grad = []
    # self.connectivity = []
    # self.last_completed_index = -1

    # self.filter_radius = get_conic_radius_from_eta_e(  # type: ignore
    #     minimum_size, sigmoid_erosion
    # )

    # self.sigmoid_bias = sigmoid_biases[0]
    # self.sigmoid_biases = sigmoid_biases
    # self.sigmoid_bias_threshold = sigmoid_bias_threshold
    # self.sigmoid_threshold = sigmoid_threshold
    # self.sigmoid_erosion = sigmoid_erosion
    # self.sigmoid_dilation = 1 - sigmoid_erosion
    # self.max_evals = evals
    # self.maximum_runtime = maximum_runtime
    # self.minimum_runtime = minimum_runtime
    # self.decay_by = decay_by
    # self.apply_connectivity = False
    # self.apply_linewidth = False
    # self._apply_connectivity = False
    # self.do_connectivity = do_connectivity
    # self.connectivity_sigmoid_threshold = connectivity_sigmoid_threshold
    # self.linewidth_sigmoid_threshold = linewidth_sigmoid_threshold
    # self.use_smoothed_projection = use_smoothed_projection

    @model_validator(mode="before")
    @classmethod
    def expand_max_evals(cls, data: Any) -> Any:
        """
        Intercepts the input dictionary to handle the int | list[int] logic.
        """
        # Pydantic passes 'data' as a dictionary of the arguments provided to __init__
        if isinstance(data, dict):
            val = data.get("max_evals")  # type: ignore
            biases = data.get("sigmoid_biases")  # type: ignore

            if isinstance(val, int) and biases is not None:
                data["max_evals"] = [val] * len(biases)  # type: ignore

        return data  # type: ignore

    @model_validator(mode="after")
    def validate_lengths(self) -> OptimizationSettings:
        if len(self.max_evals) != len(self.sigmoid_biases):
            raise ValueError(
                f"Mismatch: max_evals (len {len(self.max_evals)}) "
                f"must match sigmoid_biases (len {len(self.sigmoid_biases)})"
            )

        return self

    @computed_field
    @property
    def filter_radius(self) -> float:
        return get_conic_radius_from_eta_e(  # type: ignore
            self.minimum_feature_size, self.sigmoid_erosion
        )

    @computed_field
    @property
    def sigmoid_dilation(self) -> float:
        return 1 - self.sigmoid_erosion

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
        self._apply_connectivity = val

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

    @field_serializer("weights")
    def serialize_weights(self, weights: list[npt.NDArray[np.float64]], _) -> list[str]:
        """Converts list of arrays to list of base64 strings for JSON."""
        encoded_list: list[str] = []
        for arr in weights:
            buf = io.BytesIO()
            np.save(buf, arr)
            encoded_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return encoded_list

    @field_validator("weights", mode="before")
    @classmethod
    def deserialize_weights(cls, v: Any) -> list[npt.NDArray[np.float64]]:
        """Converts base64 strings back to numpy arrays."""
        if isinstance(v, list):
            decoded_list: list[npt.NDArray[np.float64]] = []
            for s in v:  # type: ignore
                buf = io.BytesIO(base64.b64decode(s))  # type: ignore
                decoded_list.append(np.load(buf))
            return decoded_list
        return v


class SimulationSettingsBase(BaseModel, ABC):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. This version of the class is intended for internal
    use only. If you are importing from here, stop, and import from export_settings.py
    instead.
    """

    n_design_regions: int = 1
    resolution: int
    baseline_optimization_value: float | list[float] = -np.inf
    needs_baseline: bool = True
    enforce_symmetry: bool = False
    history_fname: str
    data_dir: Path

    # def __init__(
    #     self,
    #     resolution: int,
    #     history_fname: str,
    #     data_dir: str | Path,
    #     enforce_symmetry: bool = True,
    #     n_design_regions: int = 1,
    # ):
    #     super().__init__()

    #     self.resolution = resolution

    #     self.n_design_regions = n_design_regions

    #     self.enforce_symmetry = enforce_symmetry
    #     self.history_fname = history_fname
    #     self.data_dir = Path(data_dir).resolve()

    #     self.baseline_optimization_value = -np.inf
    #     self.needs_baseline = True

    @model_validator(mode="before")
    @classmethod
    def resolve_path(cls, data: Any) -> Any:
        if isinstance(data, dict):
            path = data["data_dir"]  # type: ignore

            if isinstance(path, str):
                data["data_dir"] = Path(path).resolve()

        return data  # type: ignore

    @abstractmethod
    def is_multi_region(self) -> bool:
        pass

    @abstractmethod
    def total_n(self) -> Union[list[int], int]:
        pass

    @abstractmethod
    def total_n_raw(self) -> int:
        """Total N for all design regions

        Returns:
            int: total number of variables to optimize for in all design region(s)
        """
        pass

    def calculate_normalization(self) -> None:
        if self.needs_baseline:
            val = self.normalization()
            self.baseline_optimization_value = val
            self.needs_baseline = False

    @abstractmethod
    def normalization(self) -> float | list[float]:
        """Function to generate baseline value(s) for your optimization problem
        will be automatically called at the beginning if settings.needs_baseline
        is True. The result will be stored internally and can be accessed via
        settings.baseline_optimization_value

        Returns:
            float | list[float]: "Baseline" value(s) for your optimization problem.
                E.g. power emitted by a source, ...
        """
        return 1

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

    def get_masks(self, filter_radius: float) -> list[MaskRegion] | MaskRegion | None:
        """Get the masks (forced values) for the design variables. If you have
        more than one design region and have a mask for at least one of them,
        you must supply masks for all of them (even if they are set to False
        everywhere). This function should return the masks in the same order
        as the design regions.

        Args:
            filter_radius (float): Filter radius to apply for linewidth and
                spacing constraint to make sure the masks satisfy it as well.

        Returns:
            list[MaskRegion] | MaskRegion | None: Locations where values are forced
        """
        return None

    @abstractmethod
    def weightslike_to_raw(self, obj: Any) -> RawWeightsType:
        return np.concatenate(obj)

    @abstractmethod
    def raw_to_weightslike(self, obj: Any) -> WeightsType:
        pass

    @abstractmethod
    def filter_and_project(
        self, weights: WeightsType, optimization: OptimizationSettings
    ) -> RawWeightsType:
        pass

    @abstractmethod
    def line_width_and_spacing(
        self, weights: WeightsType, optimization: OptimizationSettings
    ) -> ConstraintReturnType:
        pass

    @abstractmethod
    def connectivity_constraint(
        self, weights: WeightsType, optimization: OptimizationSettings
    ) -> ConstraintReturnType:
        pass

    @abstractmethod
    def apply_symmetry(self, weights: Any) -> WeightsType:
        return weights

    @abstractmethod
    def get_objective(
        self,
        optimization: OptimizationSettings,
    ) -> PhysicsObjective:
        return get_physics_objective(self, optimization)

    @abstractmethod
    def get_fingerprint(self) -> str:
        pass


RegistryKey = tuple[type, type]

_OBJECTIVE_REGISTRY: dict[
    RegistryKey,
    Callable[..., PhysicsObjective],
] = {}


def register_physics_objective(sim_type: type, opt_type: type):
    """
    Decorator to register a function for a specific combination of types.
    Supports inheritance-aware dispatch.
    """

    def decorator(
        func: Callable[..., PhysicsObjective],
    ):
        _OBJECTIVE_REGISTRY[(sim_type, opt_type)] = func
        return func

    return decorator


def get_physics_objective(
    settings: SimulationSettingsBase, optimization: OptimizationSettings
) -> PhysicsObjective:
    """
    The Router. Uses 'isinstance' and MRO (Method Resolution Order)
    to find the most specific registered combination.
    """
    best_func = None
    max_specificity = -1

    # We must iterate because we aren't looking for an exact dict key,
    # but the "best" matching key in the inheritance hierarchy.
    for (reg_sim_type, reg_opt_type), func in _OBJECTIVE_REGISTRY.items():
        # 1. Check if the provided objects match the registered types
        if isinstance(settings, reg_sim_type) and isinstance(
            optimization, reg_opt_type
        ):
            # 2. Calculate Specificity Score
            # We use the MRO (Method Resolution Order) index.
            # A lower index means the class is more specific (closer to the object).
            # We want to maximize 'specificity', so we use: (MRO_Length - Index)

            sim_mro = type(settings).mro()
            opt_mro = type(optimization).mro()

            # Score = (How deep is the sim_type in the MRO?) + (How deep is the opt_type in the MRO?)
            # Using (Length - Index) ensures that the subclass (Index 0) has a higher score than the base class (Index 1).
            sim_score = len(sim_mro) - sim_mro.index(reg_sim_type)
            opt_score = len(opt_mro) - opt_mro.index(reg_opt_type)

            total_specificity = sim_score + opt_score

            # 3. Keep the match with the highest specificity
            if total_specificity > max_specificity:
                max_specificity = total_specificity
                best_func = func

    if best_func is None:
        raise NotImplementedError(
            f"No physics objective registered for: ({type(settings).__name__}, {type(optimization).__name__})"
        )

    return best_func(settings, optimization)
