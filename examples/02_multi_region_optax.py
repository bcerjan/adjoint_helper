"""
Adjoint Helper
Copyright (C) 2026 Ben Cerjan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Example of setting up an optimization with multiple design regions.
The settings (the only hard part) are defined separately so they
can be reused in other examples.
"""
from multi_region_meep_example_settings import MultiRegionExampleSettings

import adjoint_helper as ah
from adjoint_helper import DesignRegion
from adjoint_helper.optimizers.optax_optimization import OptaxOptimizationSettings

# all units in um
wavelength = 1.600
resolution = 30
subInd = 1.0  # Air
matInd = 3.4  # Si
bgInd = 1  # air
focal_distance = 15


minimum_feature = 0.09

designX = 10
designY = 2.0
n_regions = 3
design_region_gap = 2

design_regions = []
# The sizes do not need to be the same for each region, but for simplicity
# they will be kept as such.
for _ in range(n_regions):
    design_regions.append(
        DesignRegion(
            x_size=designX,
            y_size=designY,
            resolution=resolution * 1,
            # Enforces horizontal symmetry -- we flipud due to the
            # ordering of the weights in numpy
            symmetry_func=lambda w: w / 2 + ah.anp.flipud(w / 2),
        )
    )

settings = MultiRegionExampleSettings(
    design_regions=design_regions,
    wavelength=wavelength,
    resolution=resolution,
    design_region_ind=matInd,
    history_fname="Multi_Region_Nlopt.out",
    enforce_symmetry=True,
    design_region_spacing=design_region_gap,
    gap_ind=subInd,
    sub_ind=subInd,
    focal_distance=focal_distance,
    data_dir="./output/optax/",
    minimum_feature_size=minimum_feature,  # "evals" assigns that number of evaluations to every bias
    maximum_runtime=2000,
)

optax_optimization = OptaxOptimizationSettings(
    simulation_settings=settings,
    # to speficify individually, use max_evals = [5, 10, 25, 5]
    evals=3,  # very few, as demonstration
)
# If history exists at the correct location (either via fpath argument or
# data_dir / history_fname in the settings) it will be loaded. If you want
# to change any of your OptimizationSettings before restarting the simulation,
# you must do that manually afterwards via history.optimization.setting = XXXX.
#
# If no history exists, it will be created based on the provided default_optimization
# (if present).
#
# If there is no history and no default is provided, you get an error.
optax_optimization.optimize()
