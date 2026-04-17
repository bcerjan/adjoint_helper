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

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import meep as mp  # type: ignore
from matplotlib.colors import Colormap
from matplotlib.pyplot import Artist  # type: ignore

from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.constraints import filter_and_project


def imshow_animation(
    frameData: npt.NDArray[np.float_], cmap: str | Colormap = "binary"
) -> animation.FuncAnimation:
    """
    Plots an animation based on the supplied framedata using imshow. Assumes
    that the frames have been reshaped correctly beforehand.

    Must call plt.show() separately to display. Note that you also need to assign
    a variable to store the result to prevent it from getting prematurely
    garbage-collected.

    Args:
        frameData: Numpy array of 3 dimensions -- first dimension is frame index,
        second and third contain image data
    """
    if frameData.ndim != 3:
        raise ValueError(f"Expected 3D array (frames, H, W), got {frameData.ndim}D")

    if len(frameData) < 1:
        raise ValueError(
            "Length of frameData < 1. Must supply at least one frame to imshow_animation"
        )
    fig, ax = plt.subplots()  # type: ignore
    im = ax.imshow(  # type: ignore
        frameData[0, :, :],
        cmap=cmap,
        interpolation="none",
    )

    def update(frame_idx: int) -> list[Artist]:
        im.set_data(frameData[frame_idx, :, :])
        return [im]

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(frameData),
        interval=50,
        blit=True,
    )
    return ani


def create_field_animation(
    settings: SimulationSettings,
    optimization: OptimizationSettings,
    end_time: float = 200,
    output_time: float = 0.5,
    output_field=mp.output_efield_x,  # type: ignore
) -> None:
    weights = optimization.weights[-1]
    weights = filter_and_project(
        weights[:],
        settings,
        optimization,
    )

    masks = settings.border_masks(optimization)

    for mask in masks:
        weights = np.where(mask.locations.flatten(), mask.value, weights)

    opt = settings.create_opt(optimization=optimization)
    opt.update_design([weights])  # type: ignore

    opt.sim.run(  # type: ignore
        mp.at_beginning(mp.output_epsilon),  # type: ignore
        mp.to_appended("field", mp.at_every(output_time, output_field)),  # type: ignore
        until=end_time,
    )


def plot_structure(
    settings: SimulationSettings,
    optimization: OptimizationSettings,
    volume: mp.Volume = mp.Volume(
        size=mp.Vector3(mp.inf, mp.inf, 0), center=mp.Vector3()
    ),
) -> None:
    masks = settings.border_masks(optimization)

    if len(optimization.weights) > 0:
        weights = optimization.weights[-1]
    else:
        weights = np.random.rand(settings.nx_design * settings.ny_design) * 0.5

    for mask in masks:
        weights = np.where(mask.locations.flatten(), mask.value, weights)  # type: ignore

    if settings.baseline_optimization_value < 0:
        settings.baseline_optimization_value = 1

    opt = settings.create_opt(optimization=optimization)
    opt.update_design([weights])  # type: ignore

    opt.plot2D(output_plane=volume)  # type: ignore
    plt.show()  # type: ignore
