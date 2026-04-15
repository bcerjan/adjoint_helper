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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import meep as mp  # type: ignore

from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.constraints import filter_and_project


def imshow_animation(frameData: np.ndarray) -> None:
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
    if len(frameData) < 1:
        raise ValueError(
            "Length of frameData < 1. Must supply at least one frame to imshow_animation"
        )
    fig = plt.figure()
    ax = plt.imshow(
        frameData[0, :, :],
        cmap="binary",
        interpolation="none",
    )

    def update(frame: int) -> None:
        ax.set_data(frameData[frame, :, :])

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=len(frameData), interval=50
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
    opt.update_design([weights])

    opt.sim.run(
        mp.at_beginning(mp.output_epsilon),
        mp.to_appended("field", mp.at_every(output_time, output_field)),
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
    opt.update_design([weights])

    opt.plot2D(output_plane=volume)
    plt.show()
