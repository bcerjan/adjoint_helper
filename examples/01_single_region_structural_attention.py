"""Adjoint Helper
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
Example of setting up an optimization with multiple design regions using the
Physics-Informed Structural Attention (AAO) backend.
The settings (the only hard part) are defined separately so they
can be reused in other examples.
"""

import numpy as np
from single_region_meep_example_settings import SingleRegionExampleSettings

from adjoint_helper import DesignRegion, OptimizationHistory
from adjoint_helper.core.defs import WeightsType
from adjoint_helper.optimizers.structural_attention import (
    StructuralAttentionOptimizationSettings,
)
from adjoint_helper.utils.digital_design_helpers import digital_circle

# all units in um

wavelength = 0.800
resolution = 20
subInd = 1.47  # SiO2
matInd = 2.01  # SiN
bgInd = 1  # air
beamSize = 1

minimum_feature = 0.08
wgWidth = 0.5
wgHeight = 0.3

# This odd number is so I can make even tiles below
designX = 5.225
designY = designX

design_region = DesignRegion(
    x_size=designX,
    y_size=designY,
    resolution=resolution * 2,
    # the layout in numpy is transposed from what we actually want,
    # so to enforce +y/-y symmetry we do a fliplr
    symmetry_func=lambda w: w / 2 + np.fliplr(w / 2),
    # For Structural Attention we need the actual index values, so you must
    # pass them in here
    index_1=matInd,
    index_0=bgInd,
)

# Now we need to specify our digital designs
design_units: list[WeightsType] = []
# These must allow for a complete tiling of the entire design region.
# It might be a good idea to structure your simulation such that the
# size of the design region is defined by these instead of the other way
# around like it is here.
design_unit_x = design_region.nx_design // 10
design_unit_y = design_region.ny_design // 10

rad = min(design_unit_x, design_unit_y) // 4

# Our digital options are full, empty, and a circle
design_units = [
    np.ones((design_unit_x, design_unit_y)),
    np.zeros((design_unit_x, design_unit_y)),
    digital_circle(design_unit_x, design_unit_y, rad),
]

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
    history_fname="Single_Region_Structural_Attention.out",
    enforce_symmetry=True,
    minimum_feature_size=minimum_feature,
    maximum_runtime=2000,
)

structural_attention_optimization = StructuralAttentionOptimizationSettings(
    design_units=[
        design_units
    ],  # this is allowed and will copy itself to all design regions if only a single list is provided
    simulation_settings=settings,
    update_fraction=0.05,  # Initial percentage of design units updated per step
    max_evals=[5],  # Run for 50 evaluations
)

# TESTING:
weights = structural_attention_optimization.simulation_settings.weights
# hist = OptimizationHistory(optimization=structural_attention_optimization)
# hist.save_to_json()
# plt.figure()
# plt.imshow(weights[0])
# plt.show()
structural_attention_optimization.optimize()
