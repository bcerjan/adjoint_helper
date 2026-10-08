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

import logging

import numpy as np

from adjoint_helper import SimulationReturn
from adjoint_helper.core.base_settings import SimulationSettings
from adjoint_helper.core.defs import (
    RawWeightsType,
    WeightsType,
    combine_objective_values,
)

logger = logging.getLogger("__name__")

type Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type MeepObjValReturn = float | list[float] | list[Array1D]
type MeepGradReturn = (
    list[list[Array2D]]
    | list[list[Array1D]]
    | list[Array2D]
    | list[Array1D]
    | Array2D
    | Array1D
)
type MeepOptReturn = tuple[MeepObjValReturn, MeepGradReturn]


def normalize_meep_weights(weights: list[WeightsType]) -> list[RawWeightsType]:
    """Converts the weights into the format that meep expects"""
    m_weights: list[np.ndarray[tuple[int], np.dtype[np.float64 | np.float32]]] = []
    for w in weights:
        m_weights.append(w.flatten())

    return m_weights


def normalize_meep_output(
    opt_out: MeepOptReturn, settings: SimulationSettings, operation: str = "mean"
) -> SimulationReturn:
    """Function to re-nest output from meep and re-organize it they way
    adjoint_helper expects. Meep tries to help by auto-squeezing results, but
    this breaks the internal handling of the return by adjoint_helper so we
    have to unsqueeze everything. We also need to re-order the results, as
    meep returns them as objective function -> design_region -> grad -> parameter/frequency
    (as a 2D array if there are more than one)

    Also will combine the objective values (if there is more than one parameter)
    using the combination method passed in. See `_OPERATION_REGISTRY` for the options.
    """

    obj_val = opt_out[0]
    if isinstance(obj_val, list) and isinstance(obj_val[0], np.ndarray):
        obj_val = combine_objective_values(obj_val, operation)  # type: ignore
    if isinstance(obj_val, (float, int)):
        obj_val = [obj_val]

    obj_grad = opt_out[1]
    arr = np.array(obj_grad)
    ndim = arr.ndim

    # Done if we have 4
    if ndim != 4:
        if ndim == 1:
            # need to expand two before and one after the single dim.
            arr = np.expand_dims(arr, axis=0)
            arr = np.expand_dims(arr, axis=0)
            arr = np.expand_dims(arr, axis=-1)

        else:
            # Sadly, we now need to figure out on which side we need to add:
            # Note that the order here matters for how we add the extra dims
            if settings.n_design_regions == 1:
                if settings.n_objective_functions != 1:
                    arr = np.expand_dims(arr, axis=1)
                else:
                    arr = np.expand_dims(arr, axis=0)

            if settings.n_objective_functions == 1:
                arr = np.expand_dims(arr, axis=0)

            if settings.n_design_parameters == 1:
                arr = np.expand_dims(arr, axis=-1)

    normalized_grad = np.transpose(arr, (1, 0, 3, 2))

    ret = (obj_val, normalized_grad)

    return ret  # type: ignore


type FDTDxOptReturn = tuple[float, dict[str, WeightsType]]
type FDTDxWeights = dict[str, WeightsType]


def normalize_fdtdx_weights(weights: list[WeightsType]) -> FDTDxWeights:
    """We import jax.numpy here so we can pass back JAX arrays for use in JIT
    functions (like our loop). We import jax here instead of at the top as
    not all of the installations will require jax."""
    import jax.numpy as jnp

    ret = {}
    for i, w in enumerate(weights):
        temp = jnp.asarray(w.copy())
        label = f"design_region{i}"
        ret[label] = jnp.expand_dims(temp, -1)
        # ret[label] = jnp.array(jnp.expand_dims(w, -1))
    return ret


def normalize_fdtdx_output(
    opt_out: FDTDxOptReturn, settings: SimulationSettings, operation: str = "mean"
) -> SimulationReturn:
    """IMPORTANT NOTE: This is currently configured to work with single FOM and single
    Parameter optimization problems. I need to come back and update it with a standard format
    for more complivated functions."""

    logger.info(
        "FDTDx auto-normalization currently only works correctly for a single FOM and"
        " a single parameter. This should be fixed soon (hopefully)."
    )

    val = float(opt_out[0])
    grad = opt_out[1]

    grad_out: list[list[list[RawWeightsType]]] = []
    for i in range(settings.n_design_regions):
        label = f"design_region{i}"
        dr_grad = np.array(grad[label])  # explicit cast to convert from JAX
        grad_out.append([[np.ravel(dr_grad)]])

    return [val], grad_out
