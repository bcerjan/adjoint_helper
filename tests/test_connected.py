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

from test_connected_settings import ConnectedCouplerSettings
from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.optimization_history import load_history
from adjoint_helper.adam_optimization import run_adam_optimization
from adjoint_helper.nlopt_optimization import run_nlopt_optimization

"""
This test demonstrates most features of the package -- it optimizes a "floating"
grating coupler for transmission into a waveguide. It forces all the pieces of
the coupler to be connected to at least one of the specified edges.

For 'adam' optimization
"""

resolution = 20


# All units in um
wavelength = 0.785
beamSize = 1
minimumFeature = 0.08

matInd = 2.05
subInd = 3.5
bgInd = 1

wgWidth = 0.5
wgHeight = 0.3


designX = 5
designY = designX

dataDir = "output/"

settings = ConnectedCouplerSettings(
    wavelength=wavelength,
    resolution=resolution,
    matInd=matInd,
    subInd=subInd,
    bgInd=bgInd,
    beamSize=beamSize,
    wgWidth=wgWidth,
    wgHeight=wgHeight,
    designX=designX,
    designY=designY,
)

# Check if previous optimization exists and re-start if it does:
hist = load_history(settings.data_dir + settings.history_fname)

if hist is None:
    optimization = OptimizationSettings(
        minimum_size=minimumFeature,
        sigmoid_biases=[4, 16, 128, 256, 4096],
        max_evals=3,  # Very few, just for example
    )
else:
    optimization = hist.optimization


settings.history_fname = "test_nlopt_floating_grating_coupler.pkl"
run_nlopt_optimization(settings, optimization)


optimization = OptimizationSettings(
    minimum_size=minimumFeature,
    sigmoid_biases=[4, 8, 24, 32, 40],  # adam seems to prefer smaller biases?
    max_evals=3,  # Very few, just for example
)

settings.history_fname = "test_adam_floating_grating_coupler.pkl"
run_adam_optimization(settings, optimization)
