# Adjoint Helper
# Copyright (C) 2026 Ben Cerjan

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["DigitalDesignRegion", "StructuralAttentionOptimizationSettings"]

import logging
from collections.abc import Callable
from typing import Annotated, Any, Self

import numpy as np
from pydantic import Field, computed_field, model_validator

from adjoint_helper.core.pydantic_defs import SolverConfig

from ..core.base_settings import (
    DesignRegion,
    OptimizationSettings,
)
from ..core.defs import (
    RawWeightsType,
    WeightsType,
)
from ..utils.util import save_fom_history, save_output

logger = logging.getLogger(__name__)

DEFAULT_BIAS = [128.0]


class DigitalDesignRegion(DesignRegion):
    design_units: list[WeightsType]
    design_unit_weights: np.ndarray[tuple[int, int], np.dtype[np.integer]] = np.array(
        []
    )

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
                out[x * x_step : x * (x_step + 1), y * y_step : y * (y_step + 1)] = (
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
                "All design units must have the same shape with a design region"
            )

        self._check_even_tile()

        return self

    @model_validator(mode="after")
    def update_design_unit_weights(self) -> Self:
        if self.design_unit_weights.size == 0:
            x = self.nx_design // self.design_unit_x
            y = self.ny_design // self.design_unit_y
            self.design_unit_weights = np.random.randint(
                0, len(self.design_units), size=(x, y)
            )
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
        arr: WeightsType,
        func: Callable[[WeightsType], float],
    ) -> WeightsType:
        (x_size, y_size) = self.get_size_in_design_units
        x_step, y_step = self.design_unit_x, self.design_unit_y
        out = np.zeros((x_size, y_size))
        for x in range(x_size):
            x_idx = x * x_step
            x_stop = x_idx + x_step
            for y in range(y_size):
                y_idx = y * y_step
                y_stop = y_idx + y_step
                out[x, y] = func(arr[x_idx:x_stop, y_idx:y_stop])

        return out

    def _aggregate_grads_over_design_units(
        self,
        gradient: WeightsType,
    ) -> WeightsType:
        """Takes 2D gradient and produces K_ij from the paper -- the aggregate of the
        gradient for each design unit inside the provided design region"""

        return self._apply_func_over_design_units(arr=gradient, func=np.sum)

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
        k_agg = self._aggregate_grads_over_design_units(gradient=gradient)
        d_max_list = []

        def _internal_func(weights: WeightsType) -> float:
            s_max = -np.inf
            d_max = -np.inf
            for idx, s in enumerate(self.design_units):
                delta_n = weights - s
                q_norm = delta_n / (np.max(np.abs(delta_n)))
                s_d = np.vdot(q_norm, k_agg)
                if s_d > s_max:
                    s_max = s_d
                    d_max = idx

            d_max_list.append(d_max)

            return s_max

        return self._apply_func_over_design_units(
            arr=self.weights, func=_internal_func
        ), d_max_list

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
    """

    digital_design_regions: list[DigitalDesignRegion]
    sigmoid_biases: list[float] = DEFAULT_BIAS
    update_fraction: float = 0.05
    attribution_maps: list[Any] = Field(default_factory=list, exclude=True, repr=False)
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
    def check_bias_and_evals(cls, data: Any) -> Any:
        if isinstance(data, dict):
            evals = data.get("max_evals", 50)
            biases = data.get("sigmoid_biases", DEFAULT_BIAS)

            if len(evals) > 1 or len(biases) > 1:
                raise ValueError(
                    "Evals and biases must be lists of length 1"
                    f" got {len(evals)} (evals) and {len(biases)} (biases)"
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def create_digital_regions(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "digital_design_regions" in data:
            return data

        design_regions = data.get("design_regions")
        design_units = data.pop("design_units", None)

        if design_units is not None and design_regions is not None:
            if len(design_regions) != len(design_units) and len(design_units) != 1:
                raise ValueError(
                    "Must have either the same number of design regions as"
                    " sets of design units or a single set of design units to share them"
                )
            digital_drs = []
            for dr, du in zip(design_regions, design_units):
                digital_drs.append(
                    DigitalDesignRegion.from_design_region(dr, design_units=du)
                )
            data["digital_design_regions"] = digital_drs

        return data

    def optimize(self) -> list[WeightsType]:
        """
        Runs the optimization using physics-informed structural attention and stores results.
        """
        self.simulation_settings.calculate_normalization()

        # Initial design weights.
        self.simulation_settings.apply_masks()
        self.simulation_settings.apply_symmetry()
        weights: RawWeightsType = np.concatenate(self.simulation_settings.weights_raw)

        # Initialize history tracking lists
        self.obj = []
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

            # Evaluate FOM and gradients
            [fom], grad = opt(weights)
            fom: float = np.clip(fom, 0, 1)

            # Invert to account for minimization:
            fom = 1 - fom
            grad[:] = -grad

            self.obj.append(fom)
            self.weights.append(
                self.simulation_settings.vector_to_arrays(weights.copy())
            )
            self.data.append(fom)

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
                a_big = (s_big - np.min(s_big)) / (np.max(s_big) - np.min(s_big))
                p_big = np.tanh(np.abs(a_big))  # type: ignore
                r_mat = np.random.uniform(size=p_big.shape)
                p_big = np.where(p_big < r_mat, 0, p_big)

                size_x, size_y = ddr.get_size_in_design_units
                alpha = self.update_fraction * (1 - fom_best)
                n_selected = np.ceil(size_x * size_y * alpha)

                partition_idxs = np.argpartition(np.ravel(p_big), kth=n_selected)

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

            fom_prev = fom

        optimal_design_weights = self.simulation_settings.filter_and_project(
            optimization=self
        )
        save_output(self, 0, binarize=True)
        save_fom_history(
            optimization=self, history_dir=self.simulation_settings.data_dir
        )

        return optimal_design_weights
