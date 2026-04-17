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

from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.mask_region import MaskRegion
import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore
import numpy as np
import numpy.typing as npt
import autograd.numpy as npa  # type: ignore


class ConnectedCouplerSettings(SimulationSettings):
    beamSize: float
    wgWidth: float
    wgHeight: float

    def __init__(
        self,
        wavelength: float,
        resolution: int,
        matInd: float,
        subInd: float,
        bgInd: float,
        beamSize: float,
        wgWidth: float,
        wgHeight: float,
        designX: float,
        designY: float,
        enforce_symmetry: bool = True,
        connected_sides: list[int] = [0, 1, 3],  # skip 2 as that is waveguide
    ):
        self.beamSize = beamSize
        self.wgWidth = wgWidth
        self.wgHeight = wgHeight
        super().__init__(
            wavelength=wavelength,
            resolution=resolution,
            matInd=matInd,
            subInd=subInd,
            bgInd=bgInd,
            designX=designX,
            designY=designY,
            connected_sides=connected_sides,
            enforce_symmetry=enforce_symmetry,
            data_dir="./output/",
            history_fname="test_floating_grating_coupler.pkl",
        )

    def create_geometry(self) -> list[mp.GeometricObject]:
        """
        Function to create overall geometry without the design region.
        """

        padding = self.wavelength * 2
        pmlLen = self.wavelength * 2

        cell = mp.Vector3(
            2 * pmlLen + self.designX + 2 * padding,
            2 * pmlLen + self.designY + 2 * padding + self.wgWidth,
            2 * pmlLen + self.wgHeight + 2 * padding,
        )
        matMed = mp.Medium(index=self.matInd)
        backMed = mp.Medium(index=self.bgInd)
        subMed = mp.Medium(index=self.subInd)

        return [
            # Waveguide
            mp.Block(
                size=mp.Vector3(pmlLen + padding, self.wgWidth, self.wgHeight),
                center=mp.Vector3(-(pmlLen + padding + self.designX) / 2, 0, 0),
                material=matMed,
            ),
            # Supported region:
            mp.Block(
                size=mp.Vector3(cell.x, cell.y, self.wgHeight),
                center=mp.Vector3(padding + pmlLen, 0, 0),
                material=matMed,
            ),
            # Support:
            mp.Block(
                size=cell,
                center=mp.Vector3(
                    padding + pmlLen, 0, -(cell.z / 2 + self.wgHeight / 2)
                ),
                material=subMed,
            ),
            # Air gap below design region:
            mp.Block(
                size=mp.Vector3(self.designX, self.designY, cell.z),
                center=mp.Vector3(0, 0, -(cell.z / 2 + self.wgHeight / 2)),
                material=backMed,
            ),
        ]

    def apply_symmetry(self, weights: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        weights = weights.reshape(self.nx_design, self.ny_design)
        # we flip LR because numpy uses a different ordering from meep
        return npa.ravel((npa.fliplr(weights) + weights) / 2)  # type: ignore

    def border_masks(self, optimization: OptimizationSettings):
        """Return border masks for the design region.

        The masks are used to prevent violations on constraints on the
        minimum feature size at the boundaries of the design region.

        Args:
        optimization: OptimizationSettings containing optimization parameters

        Returns:
        List of masks for fixed region on the edges (including waveguide port) and forced
        empty regions (adjacent to wavguide port)
        """
        x_grid = np.linspace(
            -self.designX / 2,
            self.designX / 2,
            self.nx_design,
        )
        y_grid = np.linspace(
            -self.designY / 2,
            self.designY / 2,
            self.ny_design,
        )
        xy_grid_x, xy_grid_y = np.meshgrid(
            x_grid,
            y_grid,
            sparse=True,
            indexing="ij",
        )

        left_waveguide_port = (
            xy_grid_x <= -self.designX / 2 + optimization.filter_radius
        ) & (np.abs(xy_grid_y) <= self.wgWidth / 2)

        empty_side = (xy_grid_x <= -self.designX / 2 + optimization.filter_radius) & (
            np.abs(xy_grid_y) > self.wgWidth / 2
        )

        border_mask = (
            left_waveguide_port
            | (xy_grid_x >= self.designX / 2 - optimization.filter_radius)
            | (xy_grid_y <= -self.designY / 2 + optimization.filter_radius)
            | (xy_grid_y >= self.designY / 2 - optimization.filter_radius)
        )

        # This removes the corners on the edge with the waveguide. They either need
        # to be empty or full, and I've chosen empty.
        # Note that this needs to be "False" and not "0" because 0 turns it into a
        # float array (apparently...). "True" would be used if you wanted those
        # corners to be solid
        border_mask = np.where(empty_side, False, border_mask)

        return [
            MaskRegion(locations=border_mask, value=1),
            MaskRegion(locations=empty_side, value=0),
        ]

    def create_opt(self, optimization: OptimizationSettings) -> mpa.OptimizationProblem:
        """
        Creates an mpa Optimization Problem for optimizing a grating coupler mated to
        a waveguide. Works assuming the waveguide is in the x-y plane and exits the
        simulation in the -x direction (left).

        Args:
        settings: SimulationSettings containing the geometrical parameters,
                    will be modified in-place if normalization run is needed
        optimization: OptimizationSettings containing optimization parameters

        """
        padding = self.wavelength * 2
        pmlLen = self.wavelength * 2

        cell = mp.Vector3(
            2 * pmlLen + self.designX + 2 * padding,
            2 * pmlLen + self.designY + 2 * padding + self.wgWidth,
            2 * pmlLen + self.wgHeight + 2 * padding,
        )

        symmetries: list[mp.Symmetry] = []

        if self.enforce_symmetry:
            symmetries = [mp.Mirror(direction=mp.Y_DIR, phase=-1)]  # type: ignore

        pmlLayers = [mp.PML(thickness=pmlLen)]

        freq = 1 / self.wavelength

        df = 0.1

        src = mp.Vector3(cell.x - 2 * pmlLen, cell.y - 2 * pmlLen, 0)

        srcPos = mp.Vector3(0, 0, padding / 2)

        measurePos = mp.Vector3(
            -(pmlLen + padding / 2 + self.designX) / 2,
            0,
            0,
        )

        sources: list[mp.Source] = [
            mp.GaussianBeamSource(
                src=mp.GaussianSource(freq, fwidth=df),
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

        nxDesign = self.nx_design
        nyDesign = self.ny_design

        matGrid = mp.MaterialGrid(
            mp.Vector3(nxDesign, nyDesign, 0),
            backMed,
            matMed,
            weights=np.ones((nxDesign, nyDesign)) * 0.5,
            beta=0,
            do_averaging=False,
            damping=0,
        )

        matGridRegion = mpa.DesignRegion(
            matGrid,  # type: ignore
            volume=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(self.designX, self.designY, self.wgHeight),
            ),
        )
        geometry = self.create_geometry()

        if self.needs_baseline:
            # We need a normalization run:
            sim = mp.Simulation(
                resolution=self.resolution,
                default_material=backMed,
                cell_size=cell,
                sources=sources,
                geometry=geometry,
                boundary_layers=pmlLayers,
                symmetries=symmetries,
            )

            fr = mp.FluxRegion(
                center=mp.Vector3(srcPos.x, srcPos.y, srcPos.z - self.wavelength / 2),
                size=mp.Vector3(cell.x, cell.y, 0),
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
            self.baseline_optimization_value = empty_flux[0]
            sim.reset_meep()

            # Only need baseline once
            self.needs_baseline = False

        geometry.append(
            mp.Block(
                center=matGridRegion.center, size=matGridRegion.size, material=matGrid
            ),
        )

        sim = mp.Simulation(
            resolution=self.resolution,
            default_material=backMed,
            cell_size=cell,
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
            eig_parity=mp.ODD_Z,
        )

        obList: list[mpa.ObjectiveQuantity] = [te0]

        # Mode strength divided by input flux
        def J(alpha: float) -> float:
            return npa.abs(alpha / self.baseline_optimization_value)  # type: ignore

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=[J],
            objective_arguments=obList,
            design_regions=[matGridRegion],
            frequencies=[freq],
            decay_by=optimization.decay_by,
            maximum_run_time=optimization.maximum_runtime,
        )

        return opt
