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
from ..core.base_settings import OptimizationSettings, SimulationSettings
from ..core.optimization_history import OptimizationHistory
from .logging_utils import am_mpi_master


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

    # TODO: Should figure output be moved into the DesignRegion also?
    # Probably?
    if am_mpi_master():
        for i, w in enumerate(s_weights):
            fig, ax = plt.subplots()  # type: ignore
            ax.imshow(  # type: ignore
                w.T,
                cmap="binary",
                interpolation="none" if not binarize else "spline36",
                alpha=1.0,
            )
            ax.set_axis_off()

            settings.data_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(  # type: ignore
                settings.data_dir
                / f"optimal_design_beta{sigmoid_bias if not binarize else 'inf'}_region{i}.png",
                dpi=150,
                bbox_inches="tight",
            )

        # Save the final (unmapped) design as a 2D array in CSV format
        settings.output_csv_all(optimization=optimization, binarize=binarize)

        hist = OptimizationHistory(settings=settings, optimization=optimization)
        hist.save_to_json()


def save_fom_history(optimization: OptimizationSettings, history_dir: Path) -> None:
    if am_mpi_master():
        history_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()  # type: ignore
        plt.plot(optimization.data, "o-")  # type: ignore
        # plt.yscale("log")  # type: ignore
        plt.grid(True)  # type: ignore
        plt.xlabel("Iteration")  # type: ignore
        plt.ylabel("FOM")  # type: ignore
        plt.savefig(history_dir / "FOM.png")  # type: ignore
