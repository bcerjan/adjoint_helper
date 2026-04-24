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

import meep as mp  # type: ignore
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.optimization_history import OptimizationHistory
from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.constraints import filter_and_project


def add_flux_box(
    size: mp.Vector3,
    center: mp.Vector3 = mp.Vector3(),
) -> list[mp.FluxRegion]:
    """
    Adds 6 flux regions to create a box around the center point with the given
    size. Specifies the weight parameter on each assuming we care about flux
    leaving the box (and not entering)
    """

    out: list[mp.FluxRegion] = []

    top = mp.FluxRegion(
        center=center + mp.Vector3(0, 0, size.z / 2),
        size=mp.Vector3(size.x, size.y, 0),
    )

    bot = mp.FluxRegion(
        center=center - mp.Vector3(0, 0, size.z / 2),
        size=mp.Vector3(size.x, size.y, 0),
        weight=-1,
    )

    x_r = mp.FluxRegion(
        center=center + mp.Vector3(size.x / 2, 0, 0),
        size=mp.Vector3(0, size.y, size.z),
    )

    x_l = mp.FluxRegion(
        center=center + mp.Vector3(-size.x / 2, 0, 0),
        size=mp.Vector3(0, size.y, size.z),
        weight=-1,
    )

    y_b = mp.FluxRegion(
        center=center + mp.Vector3(0, size.y / 2, 0),
        size=mp.Vector3(size.x, 0, size.z),
    )

    y_f = mp.FluxRegion(
        center=center + mp.Vector3(0, -size.y / 2, 0),
        size=mp.Vector3(size.x, 0, size.z),
        weight=-1,
    )

    out.extend([top, bot, x_r, x_l, y_b, y_f])

    return out


def get_total_box_flux(fluxes: list[mp.DftFlux]) -> float:
    flux: float = 0

    for f in fluxes:
        temp = mp.get_fluxes(f)  # type: ignore
        flux += temp[0]  # TODO: check!

    return flux


def save_output(
    weights: npt.NDArray[np.float64],
    settings: SimulationSettings,
    optimization: OptimizationSettings,
    sigmoid_bias: float,
    history_fpath: str,
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

    if binarize:
        binarize_weights(weights, settings, optimization)

    optimal_design_weights = filter_and_project(
        weights[:],
        settings,
        optimization,
    ).reshape(settings.nx_design, settings.ny_design)

    fig, ax = plt.subplots()  # type: ignore
    ax.imshow(  # type: ignore
        optimal_design_weights,
        cmap="binary",
        interpolation="none" if not binarize else "spline36",
        alpha=1.0,
    )
    ax.set_axis_off()
    if mp.am_master():
        fig.savefig(  # type: ignore
            settings.data_dir
            + f"optimal_design_beta{sigmoid_bias if not binarize else 'inf'}.png",
            dpi=150,
            bbox_inches="tight",
        )
        # Save the final (unmapped) design as a 2D array in CSV format
        fname = (
            f"unmapped_design_weights_beta{sigmoid_bias}.csv"
            if not binarize
            else "binarized_design_weights.csv"
        )
        np.savetxt(
            settings.data_dir + fname,
            weights[:].reshape(settings.nx_design, settings.ny_design),
            fmt="%4.2f",
            delimiter=",",
        )

        hist = OptimizationHistory(settings=settings, optimization=optimization)
        hist.save_history(history_fpath)


def save_fom_history(
    optimization: OptimizationSettings,
    history_fpath: str,
) -> None:
    if mp.am_master():
        plt.figure()  # type: ignore
        plt.plot(optimization.data, "o-")  # type: ignore
        plt.yscale("log")  # type: ignore
        plt.grid(True)  # type: ignore
        plt.xlabel("Iteration")  # type: ignore
        plt.ylabel("FOM")  # type: ignore
        plt.savefig(history_fpath + "FOM.png")  # type: ignore


def binarize_weights(
    weights: npt.NDArray[np.float64],
    settings: SimulationSettings,
    optimization: OptimizationSettings,
) -> None:
    """
    Updates weights in-place to be binary. Note that this also sets optimization.sigmoid_bias
    to np.inf, so be sure to reset it afterwards if it should be something else.

    :param weights: Current weights to be binarized
    :type weights: npt.NDArray[np.float64]
    :param settings: Simulation Settings for these weights
    :type settings: SimulationSettings
    :param optimization: Optimization Settings for these weights
    :type optimization: OptimizationSettings
    """
    optimization.sigmoid_bias = np.inf
    weights[:] = np.round(
        np.sign(filter_and_project(weights, settings, optimization) - 0.5) / 2 + 0.5
    )
