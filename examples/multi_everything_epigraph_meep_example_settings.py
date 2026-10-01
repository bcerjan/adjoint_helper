import autograd.numpy as npa
import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore
import numpy as np

import adjoint_helper as ah
from adjoint_helper.utils.normalization_functions import (
    normalize_meep_output,
    normalize_meep_weights,
)

"""
Example of a settings class with multiple objectives to be simultaneously
optimized using a minimax objective via the epigraph formulation.


A recreation of: https://nbviewer.org/github/NanoComp/meep/blob/master/python/examples/adjoint_optimization/06-Near2Far-Epigraph.ipynb
but with two design regions (to demonstrate how that would work)

It also tries to maximize field strength in between the lenses
(to demonstrate how multiple objective functions would work)
"""


class MultiEverythingExampleSettings(ah.SimulationSettings):
    """These are parameters that are unique to your simulation and can be re-used
    for different variations (new substrate / materials / whatever)
    """

    design_region_ind: float
    focal_distance: float
    wavelengths: list[float]
    resolution: int

    @property
    def max_wavelength(self) -> float:
        return np.max(self.wavelengths)

    @property
    def frequencies(self) -> list[float]:
        return [1 / w for w in self.wavelengths]

    def _get_padding(self) -> float:
        return self.max_wavelength * 2

    def _get_pml(self) -> float:
        return 1.0 if self.max_wavelength < 2 else self.max_wavelength / 2

    def get_max_design_width(self) -> float:
        m_size = -1
        for d in self.design_regions:
            m_size = max(m_size, d.x_size)
        return m_size

    def get_max_design_height(self) -> float:
        m_size = -1
        for d in self.design_regions:
            m_size = max(m_size, d.y_size)
        return m_size

    def normalization(self) -> float | list[float]:
        # No normalization needed for this objective
        return super().normalization()

    def create_sim(
        self, optimization: ah.OptimizationSettings
    ) -> mpa.OptimizationProblem:
        """
        Creates an mpa Optimization Problem for our dual-metalens single-wavelength
        scenario

        Args:
        settings: SimulationSettings containing the geometrical parameters,
                    will be modified in-place if normalization run is needed
        optimization: OptimizationSettings containing optimization parameters

        """

        padding = self._get_padding()
        pmlLen = self._get_pml()

        cell = mp.Vector3(
            2 * pmlLen + self.get_max_design_width(),
            2 * pmlLen + 2 * padding + 2 * self.get_max_design_height(),
        )

        symmetries: list[mp.Symmetry] = []

        if self.enforce_symmetry:
            symmetries = [mp.Mirror(direction=mp.X)]  # type: ignore

        pmlLayers = [mp.PML(thickness=pmlLen)]

        freqs = self.frequencies

        median_f = np.mean(freqs)
        max_f = np.max(freqs)
        min_f = np.min(freqs)
        df = max_f - min_f

        src_size = mp.Vector3(self.get_max_design_width(), 0)

        srcPos = mp.Vector3(
            0, -cell.y / 2 + pmlLen + padding / 2 - self.max_wavelength / 2
        )

        measurePos = mp.Vector3(
            0,
            cell.y / 2 - pmlLen - padding / 2 + self.max_wavelength / 2,
        )

        sources: list[mp.Source] = [
            mp.Source(
                src=mp.GaussianSource(frequency=median_f, fwidth=df),
                component=mp.Ez,  # type: ignore
                center=srcPos,
                size=src_size,
            )
        ]

        matMed = mp.Medium(index=self.design_region_ind)

        matgrid_regions: list[mpa.DesignRegion] = []

        geometry = []
        dr = self.design_regions[0]
        matGrid = mp.MaterialGrid(
            mp.Vector3(dr.nx_design, dr.ny_design),
            mp.air,
            matMed,
            weights=np.ones((dr.nx_design, dr.ny_design)) * 0.5,
            beta=0,
            do_averaging=optimization.use_epsavg,
            damping=optimization.use_damping,
            grid_type="U_MEAN",
        )

        matGridRegion = mpa.DesignRegion(
            matGrid,  # type: ignore
            volume=mp.Volume(
                center=mp.Vector3(0, -dr.y_size),
                size=mp.Vector3(dr.x_size, dr.y_size),
            ),
        )
        matGrid2 = mp.MaterialGrid(
            mp.Vector3(dr.nx_design, dr.ny_design),
            mp.air,
            matMed,
            weights=np.ones((dr.nx_design, dr.ny_design)) * 0.5,
            beta=0,
            do_averaging=optimization.use_epsavg,
            damping=optimization.use_damping,
            grid_type="U_MEAN",
        )

        matGridRegion2 = mpa.DesignRegion(
            matGrid2,  # type: ignore
            volume=mp.Volume(
                center=mp.Vector3(0, dr.y_size),
                size=mp.Vector3(dr.x_size, dr.y_size),
            ),
        )

        matgrid_regions.append(matGridRegion)
        matgrid_regions.append(matGridRegion2)

        geometry.append(
            mp.Block(
                center=matGridRegion.center,
                size=matGridRegion.size,
                material=matGrid,
            ),
        )

        geometry.append(
            mp.Block(
                center=matGridRegion2.center,
                size=matGridRegion2.size,
                material=matGrid2,
            ),
        )
        sim = mp.Simulation(
            resolution=self.resolution,
            default_material=mp.air,
            cell_size=cell,
            sources=sources,
            geometry=geometry,
            boundary_layers=pmlLayers,
            symmetries=symmetries,
        )

        near_2_far_regions = [
            mp.Near2FarRegion(
                center=measurePos,
                size=mp.Vector3(self.get_max_design_width(), 0),
                weight=+1,
            )
        ]

        far_fields = mpa.Near2FarFields(
            sim=sim,
            Near2FarRegions=near_2_far_regions,
            far_pts=[mp.Vector3(0, self.focal_distance)],
        )

        central_e_field = mpa.FourierFields(
            sim=sim,
            volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(cell.x / 4, 0)),
            component=mp.Ez,
        )

        obList: list[mpa.ObjectiveQuantity] = [far_fields, central_e_field]

        # We negate to make this a minimization objective
        # Note that in meep every objective function must accept
        # arguments for all objective functions, even if unused
        def J1(far_field: list[list[complex]], _: np.ndarray) -> float:
            return -(npa.abs(far_field[0, :, 2]) ** 2)  # type: ignore

        def J2(_: list[list[complex]], dft_ez: np.ndarray) -> float:
            # If you do not use axis=1, you sum the response from all of the
            # frequencies (which may or may not be what you want)
            return -(npa.sum(npa.abs(dft_ez) ** 2, axis=1))  # type: ignore

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=[J1, J2],
            objective_arguments=obList,
            design_regions=matgrid_regions,
            frequencies=freqs,
            decay_by=optimization.decay_by,
            maximum_run_time=optimization.maximum_runtime,
        )

        return opt

    def create_opt(
        self, optimization: ah.OptimizationSettings
    ) -> ah.OptimizationFunction:

        opt = self.create_sim(optimization)

        def obj(
            weights: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
        ) -> ah.SimulationReturn:

            # Meep's opt wants 1D vectors of weights, but we receive a 2D array
            # from the optimization backend (always)
            m_weights = normalize_meep_weights(weights)

            val, grad = opt(m_weights)  # type: ignore

            # For common backends, there should be a similar utility function
            # to automatically normalize the output into the correct shape
            # for the rest of the library. If there isn't one for your backend,
            # feel free to write one and submit it as a PR.
            val, grad = normalize_meep_output((val, grad))

            return (val, grad)

        return obj
