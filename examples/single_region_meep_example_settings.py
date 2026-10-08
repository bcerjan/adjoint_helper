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
import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore

import adjoint_helper as ah
from adjoint_helper.utils.normalization_functions import (
    normalize_meep_output,
    normalize_meep_weights,
)  # type: ignore

"""
Example of a settings class -- in theory the only piece of code you need to write.
This example optimizes a waveguide pointed in the left direction with a variable
sized design region.

Note that the creation of the simulation is separate from creation of the objective.
This isn't necessary, but it makes it easier to do visualization stuff as you
can get the bare simulation instead of the objective (which obscures it)

"""


class SingleRegionExampleSettings(ah.SimulationSettings):
    """These are parameters that are unique to your simulation and can be re-used
    for different variations (new substrate / materials / whatever)
    """

    beamSize: float
    wgWidth: float
    wgHeight: float
    matInd: float
    subInd: float
    bgInd: float
    wavelength: float
    resolution: int  # type: ignore

    @property
    def _pml_len(self) -> float:
        return self.wavelength * 2

    @property
    def _padding(self) -> float:
        return self.wavelength * 2

    @property
    def _cell(self) -> mp.Vector3:
        return mp.Vector3(
            2 * self._pml_len + self.design_regions[0].x_size + 2 * self._padding,
            2 * self._pml_len
            + self.design_regions[0].y_size
            + 2 * self._padding
            + self.wgWidth,
            2 * self._pml_len + self.wgHeight + 2 * self._padding,
        )

    @property
    def _df(self) -> float:
        return 0.1

    def create_geometry(self) -> list[mp.GeometricObject]:
        """
        Function to create overall geometry without the design region.
        """

        matMed = mp.Medium(index=self.matInd)
        subMed = mp.Medium(index=self.subInd)

        return [
            # Waveguide
            mp.Block(
                size=mp.Vector3(
                    self._pml_len + self._padding, self.wgWidth, self.wgHeight
                ),
                center=mp.Vector3(
                    -(self._pml_len + self._padding + self.design_regions[0].x_size)
                    / 2,
                    0,
                    0,
                ),
                material=matMed,
            ),
            # Substrate:
            mp.Block(
                size=self._cell,
                center=mp.Vector3(0, 0, -(self._cell.z / 2 + self.wgHeight / 2)),
                material=subMed,
            ),
        ]

    def get_masks(self, filter_radius: float) -> ah.MaskRegion:
        """Return border masks for the design region.

        The masks are used to prevent violations on constraints on the
        minimum feature size at the boundaries of the design region.

        Returns:
        List of masks for fixed region on the edges (including waveguide port) and forced
        empty regions (adjacent to wavguide port)
        """
        x_grid = ah.anp.linspace(
            -self.design_regions[0].x_size / 2,
            self.design_regions[0].x_size / 2,
            self.design_regions[0].nx_design,
        )
        y_grid = ah.anp.linspace(
            -self.design_regions[0].y_size / 2,
            self.design_regions[0].y_size / 2,
            self.design_regions[0].ny_design,
        )
        xy_grid_x, xy_grid_y = ah.anp.meshgrid(
            x_grid,
            y_grid,
            sparse=True,
            indexing="ij",
        )

        left_waveguide_port = (
            xy_grid_x <= -self.design_regions[0].x_size / 2 + filter_radius
        ) & (ah.anp.abs(xy_grid_y) <= self.wgWidth / 2)

        border_mask = left_waveguide_port

        return ah.MaskRegion(locations=border_mask, value=1)  # type: ignore

    def normalization(self) -> float | list[float]:

        symmetries: list[mp.Symmetry] = []

        if self.enforce_symmetry:
            symmetries = [mp.Mirror(direction=mp.Y, phase=-1)]  # type: ignore

        pmlLayers = [mp.PML(thickness=self._pml_len)]

        freq = 1 / self.wavelength

        src = mp.Vector3(
            self._cell.x - 2 * self._pml_len, self._cell.y - 2 * self._pml_len, 0
        )

        srcPos = mp.Vector3(0, 0, self._padding / 2)

        sources: list[mp.Source] = [
            mp.GaussianBeamSource(
                src=mp.GaussianSource(freq, fwidth=self._df),
                center=srcPos,
                size=src,
                beam_x0=-1 * srcPos,
                beam_w0=self.beamSize,
                beam_E0=mp.Vector3(0, 1, 0),
                beam_kdir=mp.Vector3(
                    -0.09, 0, -1
                ),  # Approximately 5 deg tilt for source
            )
        ]

        backMed = mp.Medium(index=self.bgInd)

        # geometry = self.create_geometry()
        geometry = []

        sim = mp.Simulation(
            resolution=self.resolution,
            default_material=backMed,
            cell_size=self._cell,
            sources=sources,
            geometry=geometry,
            boundary_layers=pmlLayers,
            symmetries=symmetries,
        )

        fr = mp.FluxRegion(
            center=mp.Vector3(srcPos.x, srcPos.y, srcPos.z - self.wavelength / 2),
            size=mp.Vector3(self._cell.x, self._cell.y, 0),
        )
        flux = sim.add_flux(freq, 0, 1, fr)  # type: ignore

        sim.run(  # type: ignore
            until_after_sources=mp.stop_when_fields_decayed(  # type: ignore
                25,
                mp.Ey,  # type: ignore
                srcPos,
                1e-6,  # type: ignore
            ),
        )

        empty_flux = mp.get_fluxes(flux)  # type: ignore
        sim.reset_meep()

        return empty_flux[0]

    def create_sim(
        self, optimization: ah.OptimizationSettings
    ) -> mpa.OptimizationProblem:
        """
        Creates an mpa Optimization Problem for optimizing a grating coupler mated to
        a waveguide. Works assuming the waveguide is in the x-y plane and exits the
        simulation in the -x direction (left).

        Args:
        settings: SimulationSettings containing the geometrical parameters,
                    will be modified in-place if normalization run is needed
        optimization: OptimizationSettings containing optimization parameters

        """

        symmetries: list[mp.Symmetry] = []

        if self.enforce_symmetry:
            symmetries = [mp.Mirror(direction=mp.Y_DIR, phase=-1)]  # type: ignore

        pmlLayers = [mp.PML(thickness=self._pml_len)]

        freq = 1 / self.wavelength

        src = mp.Vector3(
            self._cell.x - 2 * self._pml_len, self._cell.y - 2 * self._pml_len, 0
        )

        srcPos = mp.Vector3(0, 0, self._padding / 2 + self.wgHeight)

        measurePos = mp.Vector3(
            -(self._pml_len + self._padding / 2 + self.design_regions[0].x_size) / 2,
            0,
            0,
        )

        sources: list[mp.Source] = [
            mp.GaussianBeamSource(
                src=mp.GaussianSource(freq, fwidth=self._df),
                center=srcPos,
                size=src,
                beam_x0=-1 * srcPos,
                beam_w0=self.beamSize,
                beam_E0=mp.Vector3(0, 1, 0),
                beam_kdir=mp.Vector3(
                    -0.09, 0, -1
                ),  # Approximately 5 deg tilt for source
            )
        ]

        matMed = mp.Medium(index=self.matInd)
        backMed = mp.Medium(index=self.bgInd)

        nxDesign = self.design_regions[0].nx_design
        nyDesign = self.design_regions[0].ny_design

        matGrid = mp.MaterialGrid(
            mp.Vector3(nxDesign, nyDesign, 0),
            backMed,
            matMed,
            weights=ah.anp.ones((nxDesign, nyDesign)) * 0.5,
            beta=optimization.sigmoid_bias,
            do_averaging=optimization.use_epsavg,
            damping=0.5 * 6.28 / self.wavelength if optimization.use_damping else 0,
        )

        matGridRegion = mpa.DesignRegion(
            matGrid,  # type: ignore
            volume=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(
                    self.design_regions[0].x_size,
                    self.design_regions[0].y_size,
                    self.wgHeight,
                ),
            ),
        )
        geometry = self.create_geometry()

        geometry.append(
            mp.Block(
                center=matGridRegion.center, size=matGridRegion.size, material=matGrid
            ),
        )

        sim = mp.Simulation(
            resolution=self.resolution,
            default_material=backMed,
            cell_size=self._cell,
            sources=sources,
            geometry=geometry,
            boundary_layers=pmlLayers,
            symmetries=symmetries,
        )

        # We want to maximize the power going in to this mode:
        te0 = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(
                center=measurePos,
                size=mp.Vector3(0, 2 * self.wgWidth, 2 * self.wgHeight),
            ),
            mode=1,
            forward=False,  # Not sure that this matters as it just inverts the sign, but we square later anyway
            eig_parity=mp.ODD_Y + mp.EVEN_Z,  # this is TE for meep
        )

        obList: list[mpa.ObjectiveQuantity] = [te0]

        # 1 - power to make it a minimization problem
        def J(alpha: float) -> float:
            return 1 - ah.anp.abs(alpha**2 / self.baseline_optimization_value)  # type: ignore

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=[J],
            objective_arguments=obList,
            design_regions=[matGridRegion],
            frequencies=[freq],
            decay_by=self.decay_by,
            maximum_run_time=self.maximum_runtime,
            minimum_run_time=self.minimum_runtime,
        )

        return opt

    def create_opt(
        self, optimization: ah.OptimizationSettings
    ) -> ah.OptimizationFunction:

        opt = self.create_sim(optimization=optimization)

        def obj(
            weights: list[
                ah.anp.ndarray[tuple[int, int], ah.anp.dtype[ah.anp.float64]]
            ],
        ) -> ah.SimulationReturn:
            # Meep's opt wants 1D vectors of weights, but we receive a 2D array
            # from the optimization backend (always)
            m_weights = normalize_meep_weights(weights)

            val, grad = opt(m_weights)  # type: ignore

            # For common backends, there should be a similar utility function
            # to automatically normalize the output into the correct shape
            # for the rest of the library. If there isn't one for your backend,
            # feel free to write one and submit it as a PR.
            val, grad = normalize_meep_output((val, grad), self)

            return (val, grad)

        return obj
