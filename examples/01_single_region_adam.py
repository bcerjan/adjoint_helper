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
import numpy as np
from single_region_meep_example_settings import SingleRegionExampleSettings

import adjoint_helper as ah
from adjoint_helper.optimizers.optax_optimization import OptaxOptimizationSettings

# all units in um

wavelength = 0.800
resolution = 5
subInd = 1.47  # SiO2
matInd = 2.01  # SiN
bgInd = 1  # air
beamSize = 1

minimum_feature = 0.08
wgWidth = 0.5
wgHeight = 0.3

designX = 5
designY = designX

design_region = ah.DesignRegion(
    x_size=designX,
    y_size=designY,
    resolution=resolution * 2,
    # the layout in numpy is transposed from what we actually want,
    # so to enforce +y/-y symmetry we do a fliplr
    symmetry_func=lambda w: w / 2 + np.fliplr(w / 2),
)

settings = SingleRegionExampleSettings(
    design_regions=[design_region],
    wavelength=wavelength,
    resolution=resolution,
    matInd=matInd,
    subInd=subInd,
    bgInd=bgInd,
    beamSize=beamSize,
    wgWidth=wgWidth,
    wgHeight=wgHeight,
    history_fname="Single_Region_Nlopt.out",
    enforce_symmetry=True,
    minimum_feature_size=minimum_feature,
)


"""
The defaults for Optax optimizations use the Adam optimizer and for simple
designs likely need little adjustment. 
"""
adam_optimization = OptaxOptimizationSettings(
    simulation_settings=settings,
    total_evals=5,  # very few, as demonstration.
)

adam_optimization.optimize()
