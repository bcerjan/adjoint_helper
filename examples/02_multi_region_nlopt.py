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

"""
This file contains a demonstration of a simple single-region single-objective
optimization. The settings (the only hard part) are defined separately so they
can be reused in other examples.
"""
import adjoint_helper as ah
from single_region_example_settings import SingleRegionExampleSettings

import nlopt  # type: ignore

# all units in um

wavelength = 0.600
resolution = 20
subInd = 1.47  # SiO2
matInd = 2.01  # SiN
bgInd = 1  # air
beamSize = 1

minimum_feature = 0.08
wgWidth = 0.5
wgHeight = 0.3

designX = 5
designY = designX

settings = SingleRegionExampleSettings(
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

"""
For nlopt, you are required to supply a solver selection due to how that libray
handles configuration. Specfically, the solver needs to know the number of parameters
when it is created:
"""
solver = nlopt.opt(nlopt.LD_CCSAQ, settings.total_n())  # type: ignore

nlopt_optimization = ah.NloptOptimizationSettings(
    optimizer=solver,
    minimum_size=minimum_feature,
    max_evals=2,  # very few, as demonstration
)

nlopt_optimization.optimize(settings=settings)
