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
import nlopt  # type: ignore
from single_region_fdtdx_example_settings import SingleRegionExampleSettings

import adjoint_helper as ah

# Optimizers must be imported individually,
from adjoint_helper.optimizers.nlopt_optimization import NloptOptimizationSettings

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

designX = 5
designY = designX


design_region = ah.DesignRegion(
    x_size=designX,
    y_size=designY,
    centering="cell",  # for FDTDx we need to adjust where the lattice is defined
    resolution=resolution,  # for FDTDx, this must be the same as the main grid
    # the layout in numpy is transposed from what we actually want,
    # so to enforce +y/-y symmetry we do a fliplr
    symmetry_func=lambda w: w / 2 + ah.anp.fliplr(w / 2),
)

settings = SingleRegionExampleSettings(
    design_regions=[design_region],
    wavelength=wavelength,
    resolution=resolution,
    matInd=matInd,
    subInd=subInd,
    bgInd=bgInd,
    beamSize=beamSize,
    beamTilt=2.5,
    wgWidth=wgWidth,
    wgHeight=wgHeight,
    history_fname="Single_Region_Nlopt_FDTDx.out",
    enforce_symmetry=True,
    minimum_feature_size=minimum_feature,
    minimum_runtime=0,
    maximum_runtime=100e-15,
)


"""
For nlopt, you are required to supply a solver selection due to how that libray
handles configuration. Specfically, the solver needs to know the number of parameters
when it is created:
"""
solver = nlopt.opt(nlopt.LD_CCSAQ, settings.total_n())  # type: ignore

nlopt_optimization = NloptOptimizationSettings(
    simulation_settings=settings,
    optimizer=solver,
    evals=2,  # very few, as demonstration.
)

# Because our mask depends on the simulation setup and optimization setup,
# we have to define it after we have our settings in place.
settings.design_regions[0].mask = settings.get_masks(nlopt_optimization.filter_radius)
# settings.plot_sim(nlopt_optimization)
# settings.print_usage(nlopt_optimization)
# settings.apply_masks()
# grad = np.random.uniform(size=len(settings.design_regions[0].weights_raw))
# settings.design_regions[0].backpropagate(gradient=grad, optimization=nlopt_optimization)
nlopt_optimization.optimize()
