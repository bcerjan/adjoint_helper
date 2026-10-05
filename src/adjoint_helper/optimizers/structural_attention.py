"""Adjoint Helper
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

__all__ = ["StructuralAttentionDesignRegion", "StructuralAttentionOptimizationSettings"]

import logging
from collections.abc import Callable
from typing import Annotated, Any, Self, TypeVar

import numpy as np
from pydantic import ConfigDict, Field, computed_field, model_validator

from adjoint_helper.core.pydantic_defs import SolverConfig

from ..core.base_settings import (
    DesignRegion,
    OptimizationSettings,
    SimulationSettings,
)
from ..core.defs import (
    RawWeightsType,
    WeightsType,
)
from ..core.pydantic_defs import IntArray2D
from ..utils.util import save_fom_history, save_output

logger = logging.getLogger(__name__)

DEFAULT_BIAS = [128.0]

DType = TypeVar("DType", bound=np.number)


class StructuralAttentionDesignRegion(DesignRegion):
    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, ser_json_bytes="base64"
    )

    design_units: list[WeightsType]
    design_unit_weights: IntArray2D = np.array([], dtype=int)

    @model_validator(mode="after")
    def _check_index_vals(self) -> Self:
        if self.index_0 is None or not isinstance(self.index_0, (int, float, complex)):
            raise ValueError(
                "Index 0 of design region must be numeric if using Structural Attention"
            )

        if self.index_1 is None or not isinstance(self.index_1, (int, float, complex)):
            raise ValueError(
                "Index 1 of design region must be numeric if using Structural Attention"
            )

        return self

    @classmethod
    def from_design_region(cls, design_region: DesignRegion, **child_kwargs) -> Self:
        return cls(**design_region.model_dump(), **child_kwargs)

    @computed_field
    @property
    def design_unit_x(self) -> int:
        return self.design_units[0].shape[0]

    @computed_field
    @property
    def design_unit_y(self) -> int:
        return self.design_units[0].shape[1]

    @computed_field
    @property
    def digitzed_weights(self) -> WeightsType:
        x_units = self.nx_design // self.design_unit_x
        y_units = self.ny_design // self.design_unit_y

        out = np.zeros_like(self.weights)
        x_step = self.design_unit_x
        y_step = self.design_unit_y
        for x in range(x_units):
            for y in range(y_units):
                out[x * x_step : (x + 1) * x_step, y * y_step : (y + 1) * y_step] = (
                    self.design_units[self.design_unit_weights[x, y]]
                )

        return out

    @model_validator(mode="after")
    def check_design_units(self) -> Self:
        """Check if all states are of the same size (per-design region) and that that
        size evenly tiles all design regions"""

        if len(self.design_units) <= 1:
            raise ValueError("Must supply at least 2 design unit options")

        shapes = [w.shape for w in self.design_units]
        identical = len(set(shapes)) <= 1
        if not identical:
            raise ValueError(
                "All design units must have the same shape within a design region"
            )

        self._check_even_tile()

        return self

    @model_validator(mode="after")
    def randomize_design_unit_weights(self) -> Self:
        if self.design_unit_weights.size == 0:
            x = self.nx_design // self.design_unit_x
            y = self.ny_design // self.design_unit_y
            self.design_unit_weights = np.random.randint(
                0,
                len(self.design_units),
                size=(x, y),
                dtype=int,
            )

            self.weights = self.digitzed_weights

        return self

    def _check_even_tile(self):
        """Assumes all states have same size (which is required anyway)"""

        _, i_rem = divmod(self.nx_design, self.design_unit_x)
        _, j_rem = divmod(self.ny_design, self.design_unit_y)

        if i_rem > 0:
            raise ValueError(
                f"Design units do not evenly tile X direction. Got remainder of {i_rem}"
            )

        if j_rem > 0:
            raise ValueError(
                f"Design units do not evenly tile Y direction. Got remainder of {j_rem}"
            )

    @computed_field
    @property
    def get_size_in_design_units(self) -> tuple[int, int]:
        return (
            self.nx_design // self.design_unit_x,
            self.ny_design // self.design_unit_y,
        )

    @computed_field
    @property
    def get_design_region_in_design_units(self) -> WeightsType:
        (num_x, num_y) = self.get_size_in_design_units

        out = np.zeros((num_x, num_y))
        return out

    def _apply_func_over_design_units(
        self,
        arr: np.ndarray[tuple[int, int], np.dtype[DType]],
        func: Callable[[np.ndarray[tuple[int, int], np.dtype[DType]]], float],
    ) -> np.ndarray[tuple[int, int], np.dtype[DType]]:
        """This function reduces a full-size input (arr) to the
        size of the design region in design units. E.g. a 100 x 100 weights
        with design units of size 10 would produce a 10x10 output."""
        (x_size, y_size) = self.get_size_in_design_units
        x_step, y_step = self.design_unit_x, self.design_unit_y
        out = np.zeros((x_size, y_size), dtype=arr.dtype)
        for x in range(x_size):
            x_idx = x * x_step
            x_stop = x_idx + x_step
            for y in range(y_size):
                y_idx = y * y_step
                y_stop = y_idx + y_step
                out[x, y] = func(arr[x_idx:x_stop, y_idx:y_stop])

        return out

    def _get_current_discrete_state(
        self, design_unit_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Maps the continuous weights to the nearest discrete state index and values."""
        current_states_idx = np.zeros_like(design_unit_weights, dtype=np.int32)
        for i in range(design_unit_weights.shape[0]):
            for j in range(design_unit_weights.shape[1]):
                diffs = [abs(design_unit_weights[i, j] - s) for s in self.design_units]
                current_states_idx[i, j] = np.argmin(diffs)

        current_discrete_vals = np.array(
            [
                [
                    self.design_units[current_states_idx[i, j]]
                    for j in range(design_unit_weights.shape[1])
                ]
                for i in range(design_unit_weights.shape[0])
            ]
        )
        return current_states_idx, current_discrete_vals

    def _weights_to_index(self, arr: WeightsType) -> WeightsType:
        delta_n = self.index_1 - self.index_0
        return arr * delta_n + self.index_0

    def _inner_loop_over_design_units(
        self, gradient: WeightsType
    ) -> tuple[WeightsType, list[int]]:
        """This performs the sub-loop over each design unit -- normalizes the
        refractive index change, projects the sensitivity, and selects the most optimal
        candidate transition.

        This is lines 10-16 in the supp. info pseudocode.
        """
        # TODO: Does this need to be the "real" gradient or before the projection?
        # I think it needs to be after, but sorta unclear.
        d_max_list: list[int] = []

        def _internal_func(
            combined: np.ndarray[tuple[int, int], np.dtype[DType]],
        ) -> float:
            s_max = -np.inf
            d_max = -10000000
            grad = np.imag(combined)
            weights = np.real(combined)
            for idx, s in enumerate(self.design_units):
                s_n = self._weights_to_index(s)
                w_n = self._weights_to_index(weights)
                delta_n = s_n - w_n
                max_delta_n: float = np.max(np.abs(delta_n))
                q_norm = (
                    delta_n / max_delta_n
                    if not np.isclose(0, max_delta_n)
                    else np.zeros_like(delta_n)
                )
                s_d = np.vdot(q_norm, grad)
                if s_d > s_max:
                    s_max = s_d
                    d_max = idx

            d_max_list.append(d_max)

            return s_max  # type: ignore

        combined = self.weights + 1j * gradient
        output = self._apply_func_over_design_units(
            arr=combined,
            func=_internal_func,
        )

        return np.real(output), d_max_list

    def _update_weights_at_design_unit(
        self,
        i: int | np.integer,
        j: int | np.integer,
        design_unit_idx: int,
    ) -> WeightsType:
        """Update weights in-place for this design region at the location of this design unit

        Returns a boolean array for use with the attribution matrix
        """

        x_step = self.design_unit_x
        y_step = self.design_unit_y
        x_start = i * x_step
        y_start = j * y_step
        x_stop = x_start + x_step
        y_stop = y_start + y_step
        self.weights[x_start:x_stop, y_start:y_stop] = self.design_units[
            design_unit_idx
        ]
        self.design_unit_weights[i, j] = design_unit_idx

        out = np.zeros_like(self.weights)
        out[x_start:x_stop, y_start:y_stop] = 1
        return out


class StructuralAttentionOptimizationSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to implement the
    Adjoint-Attention Optimization (AAO) backend based on physics-informed
    structural attention as described in Chao Luo et al., 2026.

    Note that this is a "digital" optimizer, so you must supply at least
    two choices for unit cells (`design_units`).

    Expects exactly one `sigmoid_bias` value and one `max_evals` value. The same
    sigmoid bias will be used for all simulations (probably you want this to be big)
    as this is digital.

    Additionally, expects that "optimal" FOM is 0, so you must either normalize
    or somehow divide to ensure that the FOM is in [0,1]. If it is not, it will
    be clipped to that range.

    0 is because we want to minimize for consistency with other optimizers

    `design_units` is a list by design region of design units for that region,
    so you could in theory use different sets of design units for different regions.
    This is an "undocumented" argument you can use when constructing this class. I don't
    have a great solution right now to get IDE's to be happy with it, but it does work (see
    the example).

    It is assumed that if you supply your own `digital_design_regions` they are in the same
    order as the `design_regions` inside the `simulation_settings` object. Probably don't make
    them by hand unless you know what you are doing.
    """

    digital_design_regions: list[StructuralAttentionDesignRegion] = []
    sigmoid_biases: list[float] = DEFAULT_BIAS
    update_fraction: float = 0.05
    attribution_maps: list[Any] = Field(default_factory=list, exclude=True, repr=False)
    beta: float = 0.9
    optimizer: Annotated[Any, Field(exclude=True, repr=False)] = None

    @computed_field
    @property
    def solver_config(self) -> SolverConfig:
        return SolverConfig(
            library="structural_attention",
            algorithm="AAO",
            params={
                "update_fraction": self.update_fraction,
            },
        )

    @model_validator(mode="before")
    @classmethod
    def create_digital_regions(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "digital_design_regions" in data:
            n_digitals = data.get("digital_design_regions", [])
            if len(n_digitals) > 0:
                return data

        settings: SimulationSettings | None = data.get("simulation_settings")
        if settings is None:
            raise ValueError("simulation_settings is required!")

        design_regions = settings.design_regions
        design_units = data.pop("design_units", None)

        if design_units is not None and design_regions is not None:
            if len(design_units) == 1:
                design_units = design_units * len(design_regions)

            if len(design_regions) != len(design_units):
                raise ValueError(
                    "Must have either the same number of design regions as"
                    " sets of design units or a single set of design units to share them"
                )

            digital_drs = []
            for dr, du in zip(design_regions, design_units):
                digital_drs.append(
                    StructuralAttentionDesignRegion.from_design_region(
                        dr, design_units=du
                    )
                )
            data["digital_design_regions"] = digital_drs

        if len(data["digital_design_regions"]) == 0:
            raise ValueError(
                "Digital Design Region(s) not created -- this could be due to"
                " a missing argument or some other misconfiguration."
                " It could also be a bug."
            )
        return data

    @model_validator(mode="after")
    def set_initial_weights(self) -> Self:
        self.update_weights_after_digitization()
        return self

    def update_weights_after_digitization(self) -> None:
        for dr, ddr in zip(
            self.simulation_settings.design_regions, self.digital_design_regions
        ):
            dr.weights = ddr.weights

    def optimize(self) -> list[WeightsType]:
        """
        Runs the optimization using physics-informed structural attention and stores results.
        """
        self.simulation_settings.calculate_normalization()

        # Initial design weights.
        self.simulation_settings.apply_masks()
        self.simulation_settings.apply_symmetry()

        # Initialize history tracking lists
        self.fom = []
        self.weights = []
        self.data = []

        # Initialize path-integral attribution maps (one per design region) to 0
        self.attribution_maps = [
            np.zeros_like(r.weights) for r in self.simulation_settings.design_regions
        ]

        total_iters = self.max_evals[0]
        self.sigmoid_bias = self.sigmoid_biases[0]

        fom_best = 0
        fom_prev = 0

        for iteration in range(total_iters):
            # Get the objective function
            opt = self.simulation_settings.get_objective(self)

            weights = np.concatenate(self.simulation_settings.weights_raw)

            # Evaluate FOM and gradients
            [fom_orig], grad = opt(weights)
            fom: float = np.clip(fom_orig, 0, 1)
            if np.any(fom != fom_orig):
                logger.warning(
                    "Clipping was required to bring FOM within [0,1]."
                    f" Max was: {np.max(fom_orig)} and min was: {np.min(fom_orig)}"
                )

            # Invert to account for minimization:
            fom = 1 - fom
            grad[:] = -grad

            self.fom.append(fom)
            self.weights.append(
                self.simulation_settings.vector_to_arrays(weights.copy())
            )

            save_output(optimization=self, sigmoid_bias=self.sigmoid_bias)

            # Split weights and gradients by design region
            grad_by_region = self.simulation_settings.vector_to_arrays(grad)

            fom_best = max(fom_best, fom)

            delta_fom = fom - fom_prev

            for dr_idx, ddr in enumerate(self.digital_design_regions):
                s_big, update_idx_list = ddr._inner_loop_over_design_units(
                    gradient=grad_by_region[dr_idx]
                )

                # TODO: Check if we need to ignore masked regions -- in theory yes,
                # but that requires that the masked region takes up an entire design
                # unit, which seems unlikely.
                denominator = np.max(s_big) - np.min(s_big)
                a_big = (
                    (s_big - np.min(s_big)) / denominator
                    if denominator is not np.isclose(0, denominator, atol=1e-9)
                    else np.zeros_like(s_big)
                )
                # TODO: Figure out where the typo(s?) are in the supplementary information.
                # Their code and text description don't agree on how p_big is defined. I think
                # what they want is tanh(beta * abs(a)) but it is super unclear given what they
                # actually wrote.
                p_big = np.tanh(self.beta * np.abs(a_big))  # type: ignore

                r_mat = np.random.uniform(size=p_big.shape)
                p_big = np.where(p_big < r_mat, 0, p_big)

                size_x, size_y = ddr.get_size_in_design_units
                alpha = self.update_fraction * (1 - fom_best)

                n_selected = int(np.ceil(size_x * size_y * alpha))

                partition_idxs = np.argpartition(np.ravel(p_big), kth=-n_selected)[
                    -n_selected:
                ]

                for idx in partition_idxs:
                    i, j = np.unravel_index(idx, s_big.shape)
                    attribution = ddr._update_weights_at_design_unit(
                        i=i,
                        j=j,
                        design_unit_idx=update_idx_list[idx],
                    )

                    if delta_fom > 0 and iteration > 0:
                        self.attribution_maps[dr_idx][attribution.astype(bool)] += (
                            delta_fom
                        )

                ddr.apply_mask()
                ddr.apply_symmetry()

            self.update_weights_after_digitization()
            fom_prev = fom

        optimal_design_weights = self.simulation_settings.filter_and_project(
            optimization=self
        )
        save_output(self, 0, binarize=True)
        save_fom_history(
            optimization=self, history_dir=self.simulation_settings.data_dir
        )

        return optimal_design_weights
