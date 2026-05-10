import adjoint_helper as ah


import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore
import numpy as np
import numpy.typing as npt
import autograd.numpy as npa  # type: ignore

"""
Example of a settings class -- in theory the only piece of code you need to write.
This example optimizes a waveguide pointed in the left direction with a variable
sized design region.

Note that the creation of the simulation is separate from creation of the objective.
This isn't necessary, but it makes it easier to do visualization stuff as you
can get the bare simulation instead of the objective (which obscures it)

"""


class SingleRegionExampleSettings(ah.SingleRegionSettings):
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
            2 * pmlLen + self.designX + 2 * padding,
            2 * pmlLen + self.designY + 2 * padding + self.wgWidth,
            2 * pmlLen + self.wgHeight + 2 * padding,
        )
        matMed = mp.Medium(index=self.matInd)
        subMed = mp.Medium(index=self.subInd)

        return [
            # Waveguide
            mp.Block(
                size=mp.Vector3(pmlLen + padding, self.wgWidth, self.wgHeight),
                center=mp.Vector3(-(pmlLen + padding + self.designX) / 2, 0, 0),
                material=matMed,
            ),
            # Substrate:
            mp.Block(
                size=cell,
                center=mp.Vector3(0, 0, -(cell.z / 2 + self.wgHeight / 2)),
                material=subMed,
            ),
        ]

    def apply_symmetry(
        self, weights: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """_summary_

        Args:
            weights (npt.NDArray[np.float64]): Weights in raw format (1D array)

        Returns:
            npt.NDArray[np.float64]: Weights in 2D with symmetry applied
        """
        weights = weights.reshape(self.nx_design, self.ny_design)
        # we flip LR because numpy uses a different ordering from meep
        return npa.ravel((npa.fliplr(weights) + weights) / 2)  # type: ignore

    def get_masks(self, filter_radius: float) -> list[ah.MaskRegion]:
        """Return border masks for the design region.

        The masks are used to prevent violations on constraints on the
        minimum feature size at the boundaries of the design region.

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

        left_waveguide_port = (xy_grid_x <= -self.designX / 2 + filter_radius) & (
            np.abs(xy_grid_y) <= self.wgWidth / 2
        )

        border_mask = left_waveguide_port

        return [
            ah.MaskRegion(locations=border_mask, value=1),
        ]

    def normalization(self) -> float | list[float]:

        padding = 2 * self.wavelength
        pmlLen = 2 * self.wavelength

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

        srcPos = mp.Vector3(0, 0, padding / 2 + self.wgHeight)

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
            beta=optimization.sigmoid_bias,
            do_averaging=optimization.use_epsavg,
            damping=0.5 * 6.28 / self.wavelength if optimization.use_damping else 0,
        )

        matGridRegion = mpa.DesignRegion(
            matGrid,  # type: ignore
            volume=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(self.designX, self.designY, self.wgHeight),
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
            eig_parity=mp.ODD_Z,
        )

        obList: list[mpa.ObjectiveQuantity] = [te0]

        # 1 / Mode strength divided by input flux (inverted for minimization)
        def J(alpha: float) -> float:
            return npa.abs(self.baseline_optimization_value / pow(alpha, 2))  # type: ignore

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

    def create_opt(self, optimization: ah.OptimizationSettings) -> ah.PhysicsObjective:

        # In this case, we need to wrap weights (a 1D numpy array) into a list
        # for an mpa.OptimizationProblem. The return type of the objective is expected
        # to be: tuple(list[float], numpy_array) where the list is the objective
        # value(s) and the numpy array is the gradients flattened into one large
        # array.
        def obj(
            weights: npt.NDArray[np.float64],
        ) -> ah.ObjectiveReturn:
            opt = self.create_sim(optimization=optimization)
            val, grad = opt([weights])

            return ([val], grad)  # type: ignore

        return obj
