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

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..core.base_settings import OptimizationSettings, SimulationSettingsBase
from ..core.optimization_history import OptimizationHistory
from ..core.defs import MaskRegion, RawWeightsType, WeightsType


def save_output(
    weights: WeightsType,
    settings: SimulationSettingsBase,
    optimization: OptimizationSettings,
    sigmoid_bias: float,
    history_fpath: str | Path,
    binarize: bool = False,
) -> None:
    """
    Stores the current optimization status and an image of the current design
    at the given path

    :param weights: Weights for the design region(s)
    :type weights: npt.NDArray[np.float64]
    :param settings: SimulationSettings object that contains the information
    :type settings: SimulationSettings
    :param optimization: OptimizationSettings for this optimization run
    :type optimization: OptimizationSettings
    :param sigmoid_bias: Current sigmoid bias for when the save ocurrs
    :type sigmoid_bias: float
    :param history_fpath: Where should the .pkl file be stored?
    :type history_fpath: str
    :param binarize: If true, force weights to be binarized to 0/1. If true,
        this also updates optimization.sigmoid_bias. Be sure to reset it
        if you intend to use it again afterwards.
    :type binarize: bool
    """
    # Save the unmapped weights and a bitmap image of the design weights

    s_weights = settings.weightslike_to_raw(weights)
    if isinstance(history_fpath, str):
        history_fpath = Path(history_fpath).resolve()

    if binarize:
        binarize_weights(s_weights, settings, optimization)

    optimal_design_weights = settings.filter_and_project(
        s_weights,
        optimization,
    )

    p_weights = settings.raw_to_weightslike(optimal_design_weights)

    if not isinstance(p_weights, list):
        p_weights = [p_weights]

    for i in range(settings.n_design_regions):
        fig, ax = plt.subplots()  # type: ignore
        ax.imshow(  # type: ignore
            p_weights[i],
            cmap="binary",
            interpolation="none" if not binarize else "spline36",
            alpha=1.0,
        )
        ax.set_axis_off()

        fig.savefig(  # type: ignore
            settings.data_dir
            / f"optimal_design_beta{sigmoid_bias if not binarize else 'inf'}.png",
            dpi=150,
            bbox_inches="tight",
        )
        # Save the final (unmapped) design as a 2D array in CSV format
        fname = (
            f"unmapped_design_weights_beta{sigmoid_bias}_region{i}.csv"
            if not binarize
            else f"binarized_design_weights_region{i}.csv"
        )
        np.savetxt(
            settings.data_dir / fname,
            p_weights[i],
            fmt="%4.2f",
            delimiter=",",
        )

    hist = OptimizationHistory(settings=settings, optimization=optimization)
    hist.save_to_json(history_fpath)


def save_fom_history(optimization: OptimizationSettings, history_fpath: str) -> None:
    plt.figure()  # type: ignore
    plt.plot(optimization.data, "o-")  # type: ignore
    plt.yscale("log")  # type: ignore
    plt.grid(True)  # type: ignore
    plt.xlabel("Iteration")  # type: ignore
    plt.ylabel("FOM")  # type: ignore
    plt.savefig(history_fpath + "FOM.png")  # type: ignore


def binarize_weights(
    weights: RawWeightsType,
    settings: SimulationSettingsBase,
    optimization: OptimizationSettings,
) -> None:
    """
    Updates weights in-place to be binary.

    :param weights: Current weights to be binarized
    :type weights: npt.NDArray[np.float64]
    :param settings: Simulation Settings for these weights
    :type settings: SimulationSettings
    :param optimization: Optimization Settings for these weights
    :type optimization: OptimizationSettings
    """
    sigmoid = optimization.sigmoid_bias
    optimization.sigmoid_bias = np.inf
    weights[:] = np.round(
        np.sign(settings.filter_and_project(weights, optimization) - 0.5) / 2 + 0.5
    )

    optimization.sigmoid_bias = sigmoid


def apply_masks(
    masks: MaskRegion | list[MaskRegion] | None,
    weights: RawWeightsType,
    multi_region: bool,
):
    if masks is not None:
        if not isinstance(masks, list):
            masks = [masks]

        if multi_region:
            locs = np.concatenate([np.ravel(m.locations) for m in masks])
            vals = np.concatenate([m.value for m in masks])

            weights[locs] = vals

        else:
            for mask in masks:
                weights[np.ravel(mask.locations)] = mask.value
