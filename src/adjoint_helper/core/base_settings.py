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

__all__ = ["SimulationSettings", "OptimizationSettings"]

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import numpy.typing as npt
from autograd import grad, tensor_jacobian_product
from autograd import numpy as npa
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)
from typing_extensions import Self

from ..vendors.meep.connectivity import constraint_connectivity
from ..vendors.meep.filters import (
    conic_filter,
    constraint_solid,
    constraint_void,
    get_conic_radius_from_eta_e,
    smoothed_projection,
    tanh_projection,
)  # type: ignore
from .defs import (
    ConstraintReturnType,
    Edge,
    MaskRegion,
    ObjectiveFunction,
    OptimizationFunction,
    RawWeightsType,
    WeightsType,
)

DEFAULT_BIASES: list[float] = [4, 8, 16, 24, 32, 40]


class OptimizationSettings(BaseModel, ABC):
    """
    Class to store necessary optimization parameters as well as optimization
    history.
    """

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, ser_json_bytes="base64"
    )

    obj: list[float] = []
    data: list[float] = []
    weights: list[list[WeightsType]] = Field(default_factory=list)  # type: ignore
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
    max_evals: list[int] = Field(default_factory=list)
    _use_epsavg: bool = False
    _use_damping: bool = True
    maximum_runtime: float = 200
    minimum_runtime: float = 0
    decay_by: float = 1e-6
    use_smoothed_projection: bool = False
    last_completed_index: int = -1
    minimum_feature_size: float = 0.05

    evals: int | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="before")
    @classmethod
    def expand_evals(cls, data: Any) -> Any:
        """
        Expands max_evals if passed as int. Update default below if you update
        them above.
        """
        if isinstance(data, dict):
            max_evals = data.get("max_evals")  # type: ignore
            evals = data.pop("evals")
            biases = data.get("sigmoid_biases", [4, 8, 16, 24, 32, 40])  # type: ignore

            target_evals = (
                evals
                if evals is not None
                else (max_evals if isinstance(max_evals, int) else None)
            )
            if target_evals is not None:
                data["max_evals"] = [target_evals] * len(biases)  # type: ignore

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
        idx = (
            self.last_completed_index + 1
            if self.last_completed_index + 1 < len(self.sigmoid_biases)
            else self.last_completed_index
        )
        return self.sigmoid_biases[idx]

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
    def optimize(self, settings: SimulationSettings) -> List[WeightsType]:
        pass

    # @field_serializer("weights")
    # def serialize_weights(self, weights: list[npt.NDArray[np.float64]], _) -> list[str]:
    #     """Converts list of arrays to list of base64 strings for JSON."""
    #     encoded_list: list[str] = []
    #     for arr in weights:
    #         buf = io.BytesIO()
    #         np.save(buf, arr)
    #         encoded_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    #     return encoded_list
    #
    # @field_validator("weights", mode="before")
    # @classmethod
    # def deserialize_weights(cls, v: Any) -> list[npt.NDArray[np.float64]]:
    #     """Converts base64 strings back to numpy arrays."""
    #     if isinstance(v, list):
    #         decoded_list: list[npt.NDArray[np.float64]] = []
    #         for s in v:  # type: ignore
    #             buf = io.BytesIO(base64.b64decode(s))  # type: ignore
    #             decoded_list.append(np.load(buf))
    #         return decoded_list
    #     return v


class DesignRegion(BaseModel):
    """Class to store information about a single design region and to perform
    operations on it (masking and symmetry). The only mandatory inputs are x_size,
    y_size, and the resolution of the design region. Everything else is optional tools
    to help manage your simulation."""

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, ser_json_bytes="base64"
    )

    x_size: float
    y_size: float
    resolution: int
    medium_0: Any = None
    medium_1: Any = None
    connected_sides: List[Edge] = []
    weights: WeightsType = np.array([])
    mask: MaskRegion | None = None
    symmetry_func: Callable[[WeightsType], WeightsType] | None = Field(
        default=None,
        description="Function that enforces symmetry on the weights."
        " Should take in weights as a 2D array and return them as the same shape."
        " Note that numpy uses a different ordering, so fliplr corresponds to"
        " a vertical flip for the weights.",
    )

    @model_validator(mode="after")
    def create_weights(self) -> Self:
        if len(self.weights) == 0:
            self.weights = np.ones((self.nx_design, self.ny_design)) * 0.5
        return self

    @computed_field
    @property
    def do_connectivity(self) -> bool:
        return len(self.connected_sides) > 0

    @computed_field
    @property
    def nx_design(self) -> int:
        return int(round(self.x_size * self.resolution)) + 1

    @computed_field
    @property
    def ny_design(self) -> int:
        return int(round(self.y_size * self.resolution)) + 1

    @computed_field
    @property
    def total_n(self) -> int:
        return self.nx_design * self.ny_design

    @computed_field
    @property
    def weights_raw(self) -> RawWeightsType:
        """Returns weights reshaped into the 'real' shape"""
        return self.weights.flatten()

    def apply_mask(self) -> None:
        if self.mask is None:
            return
        self.weights[self.mask.locations] = self.mask.value

    def apply_symmetry(self) -> None:
        """If this design region should have some kind of enforced
        symmetry, specify how to apply that here. Note that for
        most operations, numpy uses the 'wrong' direction for things,
        so e.g. if you have a left-right symmetry you specify that using
        np.flipud instead of 'lr'"""
        if self.symmetry_func is not None:
            self.weights = self.symmetry_func(self.weights)

    def unflatten_array(self, arr: RawWeightsType) -> WeightsType:
        return np.reshape(arr, (self.nx_design, self.ny_design))

    def filter_and_project(
        self, optimization: OptimizationSettings, weights: WeightsType | None = None
    ) -> WeightsType:
        """Returns the weights as a 2D array in the 'real' shape"""
        # TODO: Possible store all the stuff from optimizationsettings
        # in this class so you could (in theory) customize per design
        # region. Would also need to move constraints into this class as
        # methods.
        pure_weights = npa.array(weights if weights is not None else self.weights)
        weights_filtered = conic_filter(
            pure_weights,
            optimization.filter_radius,
            self.x_size,
            self.y_size,
            [self.resolution],
        )

        if optimization.use_epsavg:
            if optimization.use_smoothed_projection:
                weights_projected = smoothed_projection(
                    weights_filtered,
                    optimization.sigmoid_bias,
                    optimization.sigmoid_threshold,
                    self.resolution,
                )
            else:
                return weights_filtered
        else:
            weights_projected = tanh_projection(  # type: ignore
                weights_filtered,
                optimization.sigmoid_bias,
                optimization.sigmoid_threshold,
            )

        return weights_projected  # type: ignore

    def backpropagate(
        self, gradient: RawWeightsType, optimization: OptimizationSettings
    ) -> None:
        """Applies tensor_jacobian_product and updates the gradient in-place.
        expects gradient to be 1D
        """

        pure_weights = np.array(self.weights, dtype=float)
        pure_gradient = np.array(self.unflatten_array(gradient), dtype=float)
        x_size = self.x_size
        y_size = self.y_size
        res = self.resolution

        f_radius = optimization.filter_radius
        use_epsavg = bool(optimization.use_epsavg)
        use_smoothed = bool(optimization.use_smoothed_projection)
        sig_bias = float(optimization.sigmoid_bias)
        sig_thresh = float(optimization.sigmoid_threshold)

        def pure_math_mapping(w):
            weights_filtered = conic_filter(
                w,
                f_radius,
                x_size,
                y_size,
                [res],
            )

            if use_epsavg:
                if use_smoothed:
                    return smoothed_projection(
                        weights_filtered,
                        sig_bias,
                        sig_thresh,
                        res,
                    )
                else:
                    return weights_filtered
            else:
                return tanh_projection(
                    weights_filtered,
                    sig_bias,
                    sig_thresh,
                )

        vjp_func = tensor_jacobian_product(pure_math_mapping, argnum=0)

        vjp_out = vjp_func(pure_weights, pure_gradient)
        gradient[:] = vjp_out.flatten()

    def line_width_and_spacing_constraint(
        self,
        resolution: int,
        optimization: OptimizationSettings,
    ) -> ConstraintReturnType:
        """Constraint function for the minimum line width and spacing.

        Args:
            resolution (int): Resolution of the main grid
            optimization (OptimizationSettings): Optimization information

        Returns:
            Tuple[float, RawWeightsType: The value of the constraint function (a scalar) and the gradient.
        """

        # Possibly all need to be added to OptimizationSettings
        # Probably in a sub-class or something.
        a1 = 1e-3  # hyper parameter (primary)
        b1 = 0  # hyper parameter (secondary)
        # hyper parameter (constant factor and exponent)
        c0 = 1e6 * (optimization.filter_radius * 1 / resolution) ** 4

        def filter_func(
            a: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return conic_filter(  # type: ignore
                x=a,
                radius=optimization.filter_radius,
                Lx=self.x_size,
                Ly=self.y_size,
                resolution=[self.resolution],
            )

        def threshold_func(
            a: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return tanh_projection(  # type: ignore
                a, optimization.sigmoid_bias, optimization.sigmoid_threshold
            )

        def M1(a: npt.NDArray[np.float64]) -> float:
            return constraint_solid(  # type: ignore
                a, c0, optimization.sigmoid_erosion, filter_func, threshold_func, 1
            )

        def M2(a: npt.NDArray[np.float64]) -> float:
            return constraint_void(  # type: ignore
                a, c0, optimization.sigmoid_dilation, filter_func, threshold_func, 1
            )

        g1 = grad(M1)(self.weights)  # type: ignore
        g2 = grad(M2)(self.weights)  # type: ignore

        gradi = g1 + g2  # type: ignore

        t1 = (M1(self.weights) - b1) / a1
        t2 = (M2(self.weights) - b1) / a1

        return npa.max([t1, t2]), gradi.flatten()  # type: ignore

    def connectivity_constraint(
        self,
        optimization: OptimizationSettings,
    ) -> ConstraintReturnType:
        """Applies connectivity constraint. Empty list of edges signals to not apply
        the constraint.

        Args:
            design_region (DesignRegion): Instance of design region we are applying the constraint
                to.
            optimization (OptimizationSettings):OptimizationSettings containing optimization parameters

        Returns:
            Tuple[float, RawWeightsType: Result from connectivity -- negative
                indicates connected as well as the gradient
        """

        if not self.do_connectivity:
            return (-1, np.zeros(self.weights.flatten().shape))

        proj = self.filter_and_project(optimization=optimization)

        aggT = np.ones_like(proj) * np.inf
        aggGrad = np.zeros_like(proj)

        # These are the hyperparameters for our connectivity constraint. Maybe
        # should be moved into OptimizationSettings
        p = 3.0
        cond_s = 1e4
        thresh = 50

        # Need to store f0 and dJ_du per-rotation, then return the max value of f0 and
        # map the dJ_du's based on if they are connected or not yet.
        # Not sure exactly how to do that -- if positive f0(/T?) for all rotations, pick
        # largest dJ_du for that location
        # for i in range(0,4):
        for i in self.connected_sides:
            rot = npa.rot90(proj, i.value).flatten()  # type: ignore
            T, _, dJ_du = constraint_connectivity(  # type: ignore
                rot,  # type: ignore
                self.nx_design,
                1,
                self.ny_design,
                p=p,
                cond_s=cond_s,
                thresh=thresh,
            )

            # Need to un-rotate to preserve orientation
            # T = npa.rot90(T.reshape(settings.nx_design, settings.ny_design), -i)
            # dJ_du = npa.rot90(dJ_du.reshape(settings.nx_design, settings.ny_design), -i)
            T = T.reshape(self.nx_design, self.ny_design)  # type: ignore
            dJ_du = dJ_du.reshape(self.nx_design, self.ny_design)  # type: ignore

            cond = T < aggT  # type: ignore
            aggT = npa.where(cond, T, aggT)  # type: ignore
            aggGrad = npa.where(cond, dJ_du, aggGrad)  # type: ignore

        # This is copied from the mpa.constraint_connectivity code as we need the
        # FOM for our assembled temperature, not raw temp from any rotation or
        # combination of rotations
        def heat_func(x: npt.NDArray[np.float64]) -> float:
            return npa.sum(x**p) ** (1 / p) / thresh  # type: ignore

        f0 = heat_func(aggT) - 1  # type: ignore

        aggGrad = npa.squeeze(aggGrad)  # type: ignore

        temp = self.backpropagate(gradient=aggGrad, optimization=optimization)

        optimization.connectivity.append(f0)
        return f0, temp.flatten()  # type: ignore

    def get_fingerprint(self) -> str:
        core_params: dict[str, Any] = {
            "resolution": self.resolution,
            "total_n": self.total_n,
            "x_size": self.x_size,
            "y_size": self.y_size,
        }

        param_str = json.dumps(core_params, sort_keys=True)
        return hashlib.sha3_256(param_str.encode()).hexdigest()


class SimulationSettings(BaseModel, ABC):
    """
    Class to store all simulation parameters for save/load as well as passing
    to optimization algorithms. This version of the class is intended for internal
    use only. If you are importing from here, stop, and import from export_settings.py
    instead.
    """

    n_design_parameters: int = (
        1  # How many 'things' are we optimizing for? Typically frequencies
    )
    n_objective_functions: int = 1
    design_regions: List[DesignRegion]
    resolution: int
    baseline_optimization_value: float | list[float] = -np.inf
    needs_baseline: bool = True
    enforce_symmetry: bool = False
    history_fname: str
    data_dir: Path = Path("./output").resolve()

    @model_validator(mode="before")
    @classmethod
    def resolve_path(cls, data: Any) -> Any:
        if isinstance(data, dict):
            path = data.get("data_dir", "./output")  # type: ignore

            if isinstance(path, str):
                data["data_dir"] = Path(path).resolve()

        return data  # type: ignore

    @model_validator(mode="after")
    def check_design_regions(self) -> Self:
        if len(self.design_regions) == 0:
            raise ValueError("Must have at least one DesignRegion!")
        return self

    @computed_field
    @property
    def n_design_regions(self) -> int:
        return len(self.design_regions)

    @computed_field
    @property
    def weights(self) -> list[WeightsType]:
        if self.enforce_symmetry:
            self.apply_symmetry()
        self.apply_masks()

        ret = []
        for dr in self.design_regions:
            ret.append(dr.weights)
        return ret

    def is_multi_region(self) -> bool:
        return len(self.design_regions) > 1

    def total_n(self) -> int:
        """Total N for all design regions

        Returns:
            int: total number of variables to optimize for in all design region(s)
        """
        sum = 0
        for r in self.design_regions:
            sum += r.total_n
        return sum

    def calculate_normalization(self) -> None:
        if self.needs_baseline:
            val = self.normalization()
            self.baseline_optimization_value = val
            self.needs_baseline = False

    def get_empty_grads(self) -> list[list[list[RawWeightsType]]]:
        return [
            [
                [
                    np.zeros(r.weights.flatten().shape)
                    for _ in range(self.n_design_parameters)
                ]
                for _ in range(self.n_objective_functions)
            ]
            for r in self.design_regions
        ]

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
    def create_opt(self, optimization: OptimizationSettings) -> OptimizationFunction:
        """This is the "main" method for your optimization. You can use whatever
        simulation engine you like (meep, tidy3D, legume, COMSOL, Elmer, ... ) as
        long as it can produce gradients for your objective function (whatever
        that may be). The produced function should take in weights as the only
        argument and must return (objective, gradient) (as arrays). The weights
        will be `filter_and_project`'ed before they are sent to your function,
        so you should not do that internally.

        Args:
            optimization (OptimizationSettings): Everything needed for only the
            immediate simulation / optimization step. Not the auxillary stuff
            specific to each optimization method.

        Returns:
            out (OptimizationFunction): A function that takes in a list of weights per
            design region and returns objective values and gradients. The returned
            gradients must be stacked like:
                design_region[objective_function[design_parameter[1D gradient vector]]]


            For common backends, there should be a 'normalaize_xxx_output' function that
            automatically converts to the correct format, so all you need to do is call it.
            This function is from adjoint_helper.utils.normalization_functions. If there isn't
            one for your simulation backend, please write one and submit it as a PR.
        """
        pass

    def update_weights(self, new_weights: List[WeightsType]) -> None:
        if len(new_weights) != self.n_design_regions:
            raise ValueError(
                "Must supply same number of weights as there are design regions."
            )

        for i, r in enumerate(self.design_regions):
            r.weights[:, :] = new_weights[i]

    def vector_to_arrays(self, vec: RawWeightsType) -> List[WeightsType]:
        """Split the vector into pieces for each design region"""
        ret = []
        start = 0
        for r in self.design_regions:
            stop = start + r.total_n
            ret.append(r.unflatten_array(vec[start:stop]))
            start = stop

        return ret

    def arrays_to_vector(self, arrays: list[WeightsType]) -> RawWeightsType:
        ret = np.zeros(self.total_n(), dtype=arrays[-1].dtype)
        start = 0
        stop = 0
        for w in arrays:
            flat = w.flatten()
            stop = start + len(flat)
            ret[start:stop] = flat
            start = stop

        return ret

    def filter_and_project(
        self,
        optimization: OptimizationSettings,
    ) -> List[WeightsType]:
        """Filters and projects weights for all design regions.
        Returns the flattened array of weights for all of them."""

        ret = []
        for r in self.design_regions:
            ret.append(r.filter_and_project(optimization=optimization))
        return ret

    def line_width_and_spacing(
        self, optimization: OptimizationSettings
    ) -> ConstraintReturnType:
        grads = np.zeros(self.total_n())
        val = 0  # TODO: possibly make this a list. Or use inequality constraints
        # to enforce the rules for multiple design regions...
        start = 0
        for r in self.design_regions:
            stop = start + r.total_n
            v, g = r.line_width_and_spacing_constraint(
                resolution=self.resolution, optimization=optimization
            )
            val += v
            grads[start:stop] = g
            start = stop
        return (val, grads)

    def connectivity_constraint(
        self, optimization: OptimizationSettings
    ) -> ConstraintReturnType:
        grads = np.zeros(self.total_n())
        val = 0  # TODO: possibly make this a list. Or use inequality constraints
        # to enforce the rules for multiple design regions...
        start = 0
        for r in self.design_regions:
            stop = start + r.total_n
            v, g = r.connectivity_constraint(optimization=optimization)
            val += v
            grads[start:stop] = g
            start = stop
        return (val, grads)

    def apply_symmetry(self) -> None:
        """Applies symmetry to all design regions internally."""
        if self.enforce_symmetry:
            for r in self.design_regions:
                r.apply_symmetry()

    def apply_masks(self) -> None:
        """Applies masks to all design regions internally."""
        for r in self.design_regions:
            r.apply_mask()

    def binarize_weights(self, optimization: OptimizationSettings) -> List[WeightsType]:
        """Returns a binarized version of the internal weights."""

        bias = optimization.sigmoid_bias
        optimization.sigmoid_bias = np.inf
        ret = []
        for r in self.design_regions:
            ret.append(r.filter_and_project(optimization=optimization))

        optimization.sigmoid_bias = bias
        return ret

    def get_objective(
        self,
        optimization: OptimizationSettings,
    ) -> ObjectiveFunction:
        return get_objective(self, optimization)

    def get_fingerprint(self) -> str:
        core_params: dict[str, Any] = {
            "resolution": self.resolution,
            "total_n": self.total_n(),
            "n_design_regions": self.n_design_regions,
        }

        idx = 0
        for r in self.design_regions:
            core_params[f"region{idx}"] = r.get_fingerprint()

        param_str = json.dumps(core_params, sort_keys=True)
        return hashlib.sha3_256(param_str.encode()).hexdigest()


RegistryKey = tuple[type, type]

_OBJECTIVE_REGISTRY: dict[
    RegistryKey,
    Callable[..., ObjectiveFunction],
] = {}


def register_objective(sim_type: type, opt_type: type):
    """
    Decorator to register a function for a specific combination of types.
    Supports inheritance-aware dispatch.
    """

    def decorator(
        func: Callable[..., ObjectiveFunction],
    ):
        _OBJECTIVE_REGISTRY[(sim_type, opt_type)] = func
        return func

    return decorator


def get_objective(
    settings: SimulationSettings, optimization: OptimizationSettings
) -> ObjectiveFunction:

    best_func = None
    max_specificity = -1

    for (reg_sim_type, reg_opt_type), func in _OBJECTIVE_REGISTRY.items():
        if isinstance(settings, reg_sim_type) and isinstance(
            optimization, reg_opt_type
        ):
            sim_mro = type(settings).mro()
            opt_mro = type(optimization).mro()

            sim_score = len(sim_mro) - sim_mro.index(reg_sim_type)
            opt_score = len(opt_mro) - opt_mro.index(reg_opt_type)

            total_specificity = sim_score + opt_score

            if total_specificity > max_specificity:
                max_specificity = total_specificity
                best_func = func

    if best_func is None:
        raise NotImplementedError(
            f"No physics objective registered for: ({type(settings).__name__}, {type(optimization).__name__})"
        )

    return best_func(settings, optimization)
