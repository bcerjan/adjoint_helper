import adjoint_helper as ah


import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore
import numpy as np
import autograd.numpy as npa

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
    resolution: int

    def create_geometry(self) -> list[mp.GeometricObject]:
        """
        Function to create overall geometry without the design region.
        """

        padding = self.wavelength * 2
        pmlLen = self.wavelength * 2

        cell = mp.Vector3(
            2 * pmlLen + self.design_regions[0].x_size + 2 * padding,
            2 * pmlLen + self.design_regions[0].y_size + 2 * padding + self.wgWidth,
            2 * pmlLen + self.wgHeight + 2 * padding,
        )
        matMed = mp.Medium(index=self.matInd)
        subMed = mp.Medium(index=self.subInd)

        return [
            # Waveguide
            mp.Block(
                size=mp.Vector3(pmlLen + padding, self.wgWidth, self.wgHeight),
                center=mp.Vector3(
                    -(pmlLen + padding + self.design_regions[0].x_size) / 2, 0, 0
                ),
                material=matMed,
            ),
            # Substrate:
            mp.Block(
                size=cell,
                center=mp.Vector3(0, 0, -(cell.z / 2 + self.wgHeight / 2)),
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
        x_grid = np.linspace(
            -self.design_regions[0].x_size / 2,
            self.design_regions[0].x_size / 2,
            self.design_regions[0].nx_design,
        )
        y_grid = np.linspace(
            -self.design_regions[0].y_size / 2,
            self.design_regions[0].y_size / 2,
            self.design_regions[0].ny_design,
        )
        xy_grid_x, xy_grid_y = np.meshgrid(
            x_grid,
            y_grid,
            sparse=True,
            indexing="ij",
        )

        left_waveguide_port = (
            xy_grid_x <= -self.design_regions[0].x_size / 2 + filter_radius
        ) & (np.abs(xy_grid_y) <= self.wgWidth / 2)

        border_mask = left_waveguide_port

        return ah.MaskRegion(locations=border_mask, value=1)  # type: ignore

    def normalization(self) -> float | list[float]:

        padding = 2 * self.wavelength
        pmlLen = 2 * self.wavelength

        cell = mp.Vector3(
            2 * pmlLen + self.design_regions[0].x_size + 2 * padding,
            2 * pmlLen + self.design_regions[0].y_size + 2 * padding + self.wgWidth,
            2 * pmlLen + self.wgHeight + 2 * padding,
        )

        symmetries: list[mp.Symmetry] = []

        if self.enforce_symmetry:
            symmetries = [mp.Mirror(direction=mp.Y, phase=-1)]  # type: ignore

        pmlLayers = [mp.PML(thickness=pmlLen)]

        freq = 1 / self.wavelength

        df = 0.1

        src = mp.Vector3(cell.x - 2 * pmlLen, cell.y - 2 * pmlLen, 0)

        srcPos = mp.Vector3(0, 0, padding / 2)

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

        backMed = mp.Medium(index=self.bgInd)

        geometry = self.create_geometry()

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
        padding = 2 * self.wavelength
        pmlLen = 2 * self.wavelength

        cell = mp.Vector3(
            2 * pmlLen + self.design_regions[0].x_size + 2 * padding,
            2 * pmlLen + self.design_regions[0].y_size + 2 * padding + self.wgWidth,
            2 * pmlLen + self.wgHeight + 2 * padding,
        )

        symmetries: list[mp.Symmetry] = []

        if self.enforce_symmetry:
            symmetries = [mp.Mirror(direction=mp.Y_DIR, phase=-1)]  # type: ignore

        pmlLayers = [mp.PML(thickness=pmlLen)]

        freq = 1 / self.wavelength

        df = 0.1

        src = mp.Vector3(cell.x - 2 * pmlLen, cell.y - 2 * pmlLen, 0)

        srcPos = mp.Vector3(0, 0, padding / 2 + self.wgHeight)

        measurePos = mp.Vector3(
            -(pmlLen + padding / 2 + self.design_regions[0].x_size) / 2,
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

        nxDesign = self.design_regions[0].nx_design
        nyDesign = self.design_regions[0].ny_design

        matGrid = mp.MaterialGrid(
            mp.Vector3(nxDesign, nyDesign, 0),
            backMed,
            matMed,
            weights=np.ones((nxDesign, nyDesign)) * 0.5,
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
            eig_parity=mp.ODD_Y + mp.EVEN_Z,  # this is TE for meep
        )

        obList: list[mpa.ObjectiveQuantity] = [te0]

        # 1 / Mode strength divided by input flux (inverted for minimization)
        def J(alpha: float) -> float:
            return npa.abs(self.baseline_optimization_value / alpha)  # type: ignore

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

    def create_opt(
        self, optimization: ah.OptimizationSettings
    ) -> ah.OptimizationFunction:

        opt = self.create_sim(optimization=optimization)

        # In this case, we need to wrap weights (a 1D numpy array) into a list
        # for an mpa.OptimizationProblem. The return type of the objective is expected
        # to be: tuple(list[float], numpy_array) where the list is the objective
        # value(s) and the numpy array is the gradients flattened into one large
        # array.
        def obj(
            weights: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
        ) -> ah.SimulationReturn:
            m_weights = normalize_meep_weights(weights)
            val, grad = opt(m_weights)  # type: ignore

            return normalize_meep_output((val, grad))

        return obj
