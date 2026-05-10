import adjoint_helper as ah


import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore
import numpy as np
import numpy.typing as npt
import autograd.numpy as npa  # type: ignore

"""
Example of a settings class with multiple design regions.
This example optimizes a pair of spaced design regions to focus light of a single
frequency at a design distance. The input is a planewave with X polarization.
"""


class MultiRegionExampleSettings(ah.MultiRegionSettings):
    """These are parameters that are unique to your simulation and can be re-used
    for different variations (new substrate / materials / whatever)
    """

    design_region_height: float
    design_region_spacing: float
    design_region_ind: float
    gap_ind: float
    sub_ind: float
    focal_distance: float
    wavelength: float
    resolution: int

    def create_geometry(self) -> list[mp.GeometricObject]:
        """
        Function to create overall geometry without the design region.
        """

        padding = self.wavelength * 2
        pmlLen = self.wavelength * 2

        cell = mp.Vector3(
            2 * pmlLen + self.designX[0] + 2 * padding,
            2 * pmlLen + self.designY[0] + 2 * padding,
            2 * pmlLen
            + 2 * self.design_region_height
            + self.design_region_spacing
            + 2 * padding,
        )
        matMed = mp.Medium(index=self.design_region_ind)
        subMed = mp.Medium(index=self.sub_ind)

        return [
            # Gap
            mp.Block(
                size=mp.Vector3(cell.x, cell.y, self.design_region_spacing),
                center=mp.Vector3(
                    0, 0, self.design_region_height + self.design_region_height / 2
                ),
                material=matMed,
            ),
            # Lower design region (will be partially overwritten):
            mp.Block(
                size=mp.Vector3(cell.x, cell.y, self.design_region_height / 2),
                center=mp.Vector3(0, 0, 0),
                material=matMed,
            ),
            # Substrate:
            mp.Block(
                size=mp.Vector3(cell.x, cell.y, cell.z / 2),
                center=mp.Vector3(0, 0, -cell.z / 4),
                material=subMed,
            ),
        ]

    def apply_symmetry(
        self, weights: list[npt.NDArray[np.float64]]
    ) -> list[npt.NDArray[np.float64]]:
        """Applies symmetry to weights. Takes in list of 2D weights and applies
        symmetry to them. Might need to do raw_to_weightlsike and weightslike_to_raw
        around this function call.

        Args:
            weights (list[npt.NDArray[np.float64]]): List of weights per design region
                already reshapes into appropriate 2D arrays

        Returns:
            list[npt.NDArray[np.float64]]: List of 2D arrays after symmetry
        """
        ret: list[np.ndarray]
        # we flip LR because numpy uses a different ordering from meep
        return npa.ravel((npa.fliplr(weights) + weights) / 2)  # type: ignore

    def normalization(self) -> float | list[float]:

        padding = self.wavelength
        pmlLen = self.wavelength

        cell = mp.Vector3(
            2 * pmlLen + self.designX[0] + 2 * padding,
            2 * pmlLen + self.designY[0] + 2 * padding,
            2 * pmlLen
            + 2 * self.design_region_height
            + self.design_region_spacing
            + 2 * padding,
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
            mp.Source(
                src=mp.GaussianSource(freq, fwidth=df),
                center=srcPos,
                size=src,
                component=mp.Ey,
            )
        ]

        backMed = mp.Medium(index=1)

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

    def create_opt(self, optimization: ah.OptimizationSettings) -> ah.PhysicsObjective:
        """
        Creates an mpa Optimization Problem for optimizing a grating coupler mated to
        a waveguide. Works assuming the waveguide is in the x-y plane and exits the
        simulation in the -x direction (left).

        Args:
        settings: SimulationSettings containing the geometrical parameters,
                    will be modified in-place if normalization run is needed
        optimization: OptimizationSettings containing optimization parameters

        """
        padding = self.wavelength
        pmlLen = self.wavelength

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
            do_averaging=optimization.use_epsavg,
            damping=optimization.use_damping,
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

        # In this case, we need to wrap weights (a 1D numpy array) into a list
        # for an mpa.OptimizationProblem. The return type of the objective is expected
        # to be: tuple(list[float], numpy_array) where the list is the objective
        # value(s) and the numpy array is the gradients flattened into one large
        # array.
        def obj(
            weights: npt.NDArray[np.float64],
        ) -> ah.ObjectiveReturn:

            val, grad = opt([weights])  # type: ignore

            return ([val], grad)  # type: ignore

        return obj
