import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore
import numpy as np

import adjoint_helper as ah
from adjoint_helper.utils.normalization_functions import (
    normalize_meep_output,
    normalize_meep_weights,
)

"""
Example of a settings class with multiple design regions.
This example optimizes a pair of spaced design regions to focus light of a single
frequency at a design distance. The input is a planewave with linear polarization.

It is set up such that you could add more design regions and it would adjust if 
you so desired.

A recreation of: https://nbviewer.jupyter.org/github/NanoComp/meep/blob/master/python/examples/adjoint_optimization/05-Near2Far.ipynb
but with multiple design regions (to demonstrate how that would work)
"""


class MultiRegionExampleSettings(ah.SimulationSettings):
    """These are parameters that are unique to your simulation and can be re-used
    for different variations (new substrate / materials / whatever)
    """

    design_region_spacing: float
    design_region_ind: float
    gap_ind: float
    sub_ind: float
    focal_distance: float
    wavelength: float
    resolution: int  # type: ignore

    def _get_padding(self) -> float:
        return self.wavelength * 2

    def _get_pml(self) -> float:
        return 1.0

    def get_max_design_width(self) -> float:
        max_w = -1
        for d in self.design_regions:
            max_w = max(max_w, d.x_size)
        return max_w

    def get_total_design_height(self) -> float:
        design_y = np.sum([d.y_size for d in self.design_regions])
        return design_y + self.design_region_spacing * (self.n_design_regions - 1)

    def get_design_bottom_pos(self) -> float:
        return -(self.get_total_design_height() / 2)

    def get_design_top_pos(self) -> float:
        return self.get_design_bottom_pos() + self.get_total_design_height()

    def create_geometry(self) -> list[mp.GeometricObject]:
        """
        Function to create overall geometry without the design regions.
        """

        padding = self._get_padding()
        pmlLen = self._get_pml()

        cell = mp.Vector3(
            2 * pmlLen + self.get_max_design_width(),
            2 * pmlLen + self.get_total_design_height() + 2 * padding,
        )

        subMed = mp.Medium(index=self.sub_ind)

        h = pmlLen + padding + self.get_total_design_height()

        return [
            # Chunk of substrate that fills the lower space up to the bottom of
            # the second design region. The design regions will overwrite this:
            mp.Block(
                size=mp.Vector3(cell.x, h, 0),
                center=mp.Vector3(
                    0,
                    -cell.y / 2 + h / 2,
                    0,
                ),
                material=subMed,
            ),
        ]

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
            2 * pmlLen + self.get_total_design_height() + 2 * padding,
        )

        symmetries: list[mp.Symmetry] = []

        if self.enforce_symmetry:
            symmetries = [mp.Mirror(direction=mp.X)]  # type: ignore

        pmlLayers = [mp.PML(thickness=pmlLen)]

        freq = 1 / self.wavelength

        df = 0.2 * freq

        src_size = mp.Vector3(self.get_max_design_width(), 0)

        srcPos = mp.Vector3(0, self.get_design_bottom_pos() - self.wavelength)

        measurePos = mp.Vector3(
            0,
            self.get_design_top_pos() + self.wavelength,
        )

        sources: list[mp.Source] = [
            mp.Source(
                src=mp.GaussianSource(freq, fwidth=df),
                component=mp.Ez,  # type: ignore
                center=srcPos,
                size=src_size,
            )
        ]

        matMed = mp.Medium(index=self.design_region_ind)
        backMed = mp.Medium(index=self.sub_ind)

        matgrid_regions: list[mpa.DesignRegion] = []

        geometry = self.create_geometry()
        total_height = -cell.y / 2 + pmlLen + padding
        for dr in self.design_regions:
            matGrid = mp.MaterialGrid(
                mp.Vector3(dr.nx_design, dr.ny_design),
                backMed,
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
                    center=mp.Vector3(0, total_height + dr.y_size / 2),
                    size=mp.Vector3(dr.x_size, dr.y_size),
                ),
            )

            matgrid_regions.append(matGridRegion)

            total_height += dr.y_size + self.design_region_spacing

            geometry.append(
                mp.Block(
                    center=matGridRegion.center,
                    size=matGridRegion.size,
                    material=matGrid,
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

        obList: list[mpa.ObjectiveQuantity] = [far_fields]

        # We negate to make this a minimization objective
        def J(far_field: list[list[complex]]) -> float:
            return -ah.anp.abs(far_field[0, :, 2] ** 2)  # type: ignore

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=[J],
            objective_arguments=obList,
            design_regions=matgrid_regions,
            frequencies=[freq],
            decay_by=self.decay_by,
            maximum_run_time=self.maximum_runtime,
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
            val, grad = normalize_meep_output((val, grad), self)

            return (val, grad)

        return obj
