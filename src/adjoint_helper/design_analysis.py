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

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from adjoint_helper.constraints import filter_and_project
from adjoint_helper.optimization_history import load_history
from adjoint_helper.visualization import imshow_animation


hist = load_history("./temp")


plt.figure()
plt.plot(hist.optimization.obj)

# Prepare data for animation:
weights = hist.optimization.weights
num_weights = len(weights)

frames = np.ones((num_weights, hist.settings.nx_design, hist.settings.ny_design))

# print(np.max(weights[28]))
for i in range(num_weights):
    frames[i, :, :] = filter_and_project(
        weights[i], hist.settings, hist.optimization
    ).reshape(hist.settings.nx_design, hist.settings.ny_design)
# print(frames.shape)

# plt.imshow(frames[29] - frames[28], cmap="binary", interpolation="none",)
ani = imshow_animation(frames)

# # Show everything
plt.show()
