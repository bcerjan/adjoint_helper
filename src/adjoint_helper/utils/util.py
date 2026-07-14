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
from ..core.base_settings import OptimizationSettings, SimulationSettings
from ..core.optimization_history import OptimizationHistory
from ..core.defs import MaskRegion, RawWeightsType, WeightsType


def save_output(
    settings: SimulationSettings,
    optimization: OptimizationSettings,
    sigmoid_bias: float,
    binarize: bool = False,
) -> None:
    """
    Stores the current optimization status and an image of the current design
    at the given path

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

    settings.apply_masks()
    settings.apply_symmetry()
    s_weights = []
    if binarize:
        s_weights = settings.binarize_weights(optimization)
    else:
        s_weights = settings.filter_and_project(optimization=optimization)

    for i, w in enumerate(s_weights):
        fig, ax = plt.subplots()  # type: ignore
        ax.imshow(  # type: ignore
            w.T,
            cmap="binary",
            interpolation="none" if not binarize else "spline36",
            alpha=1.0,
        )
        ax.set_axis_off()

        fig.savefig(  # type: ignore
            settings.data_dir
            / f"optimal_design_beta{sigmoid_bias if not binarize else 'inf'}_region{i}.png",
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
            w,
            fmt="%4.2f",
            delimiter=",",
        )

    hist = OptimizationHistory(settings=settings, optimization=optimization)
    hist.save_to_json(settings.data_dir / settings.history_fname)


def save_fom_history(optimization: OptimizationSettings, history_fpath: str) -> None:
    plt.figure()  # type: ignore
    plt.plot(optimization.data, "o-")  # type: ignore
    plt.yscale("log")  # type: ignore
    plt.grid(True)  # type: ignore
    plt.xlabel("Iteration")  # type: ignore
    plt.ylabel("FOM")  # type: ignore
    plt.savefig(history_fpath + "FOM.png")  # type: ignore
