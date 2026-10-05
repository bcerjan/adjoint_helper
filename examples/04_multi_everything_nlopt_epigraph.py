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
import autograd.numpy as npa
import nlopt
from multi_everything_epigraph_meep_example_settings import (
    MultiEverythingExampleSettings,
)

from adjoint_helper import DesignRegion, OptimizationHistory
from adjoint_helper.optimizers.nlopt_epigraph_optimization import (
    NloptEpigraphOptimizationSettings,
)

# all units in um
wavelength = 1.600
resolution = 30
subInd = 1.0  # Air
matInd = 3.4  # Si
bgInd = 1  # air
focal_distance = 15
wavelengths = [1.5, 1.55, 1.6]

minimum_feature = 0.09

designX = 15
designY = 2.0
n_regions = 4
design_region_gap = 2

# This example assumes both design regions are identical --
# for different sizes/shapes look at the 02 examples
design_regions = [
    DesignRegion(
        x_size=designX,
        y_size=designY,
        resolution=resolution * 1,
        # Enforces horizontal symmetry -- we flipud due to the
        # ordering of the weights in numpy
        # We also need to use the npa version (instead of np)
        # as we backpropagate through this function
        symmetry_func=lambda w: w / 2 + npa.flipud(w / 2),
    )
    for _ in range(n_regions)
]

settings = MultiEverythingExampleSettings(
    design_regions=design_regions,
    design_region_spacing=design_region_gap,
    wavelengths=wavelengths,
    n_design_parameters=len(wavelengths),
    resolution=resolution,
    design_region_ind=matInd,
    history_fname="Multi_Parameter_Nlopt.out",
    enforce_symmetry=True,
    focal_distance=focal_distance,
    data_dir="./output/multi_parameter",
    minimum_feature_size=minimum_feature,  # "evals" assigns that number of evaluations to every bias
    maximum_runtime=2000,
)

solver = nlopt.opt(nlopt.LD_CCSAQ, settings.total_n() + 1)  # for epigraph variable

nlopt_optimization = NloptEpigraphOptimizationSettings(
    simulation_settings=settings,
    n_objectives=settings.n_objectives,
    optimizer=solver,
    # to speficify individually, use max_evals = [5, 10, 25, 5]
    evals=3,  # very few, as demonstration
    sigmoid_biases=[4, 8, 16, 32, 64, 128],
)
# If history exists at the correct location (either via fpath argument or
# data_dir / history_fname in the settings) it will be loaded. If you want
# to change any of your OptimizationSettings before restarting the simulation,
# you must do that manually afterwards via history.optimization.setting = XXXX.
#
# If no history exists, it will be created based on the provided default_optimization
#
# Other options are `OptimizationHistory.load_if_exists()` and
# `OptimizationHistory.load_from_json()`,
# history = OptimizationHistory.load_or_create(
#     settings=settings, default_optimization=nlopt_optimization
# )
history = OptimizationHistory(optimization=nlopt_optimization)

# If we loaded the history, we need to re-set the symmetry functions
# as they are not stored (so technically I didn't need to write them above)
for dr in history.settings.design_regions:
    dr.symmetry_func = lambda w: w / 2 + npa.flipud(w / 2)

# `optimize` both starts and restarts the process
history.optimize()
