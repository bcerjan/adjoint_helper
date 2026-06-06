from inspect import isclass
from typing import List
import adjoint_helper as ah


import meep as mp  # type: ignore
import meep.adjoint as mpa  # type: ignore
import numpy as np
import numpy.typing as npt
import autograd.numpy as npa  # type: ignore

"""
Example of a settings class with multiple design regions.
This example optimizes a pair of spaced design regions to focus light of a single
frequency at a design distance. The input is a planewave with linear polarization.

It is set up such that you could add more design regions and it would adjust if 
you so desired.

A recreation of: https://nbviewer.jupyter.org/github/NanoComp/meep/blob/master/python/examples/adjoint_optimization/05-Near2Far.ipynb
but with a second design region (to demonstrate how that would work)
"""


class MultiRegionExampleSettings(ah.MultiRegionSettings):
    """These are parameters that are unique to your simulation and can be re-used
    for different variations (new substrate / materials / whatever)
    """

    design_region_spacing: float
    design_region_ind: float
    gap_ind: float
    sub_ind: float
    focal_distance: float
    wavelength: float
    resolution: int

    def _get_pading(self) -> float:
        return self.wavelength * 3

    def _get_pml(self) -> float:
        return 1.0

    def get_max_design_width(self) -> float:
        return max(self.designX)

    def get_total_design_height(self) -> float:
        return np.sum(self.designY[:-1]) + self.design_region_spacing * (
            self.n_design_regions - 1
        )

    def create_geometry(self) -> list[mp.GeometricObject]:
        """
        Function to create overall geometry without the design regions.
        """

        padding = self._get_pading()
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

    def apply_symmetry(
        self, weights: list[npt.NDArray[np.float64]]
    ) -> list[npt.NDArray[np.float64]]:
        """Applies symmetry to weights. Takes in list of 2D weights and applies
        symmetry to them. Should be sorted into design regions already.

        Args:
            weights (list[npt.NDArray[np.float64]]): List of weights per design region
                already reshaped into appropriate 2D arrays

        Returns:
            list[npt.NDArray[np.float64]]: List of 2D arrays after symmetry
        """
        # we flip UD because numpy uses a different ordering from meep
        ret: list[np.ndarray] = []
        for i in range(self.n_design_regions):
            ret.append((npa.flipud(weights[i]) + weights[i]) / 2)  # type: ignore

        return ret

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

        padding = self._get_pading()
        pmlLen = self._get_pml()

        cell = mp.Vector3(
            2 * pmlLen + self.get_max_design_width(),
            2 * pmlLen + self.get_total_design_height() + 2 * padding,
        )

        symmetries: list[mp.Symmetry] = []

        if self.enforce_symmetry:
            symmetries = [mp.Mirror(direction=mp.X_DIR)]  # type: ignore

        pmlLayers = [mp.PML(thickness=pmlLen)]

        freq = 1 / self.wavelength

        df = 0.2

        src = mp.Vector3(self.get_max_design_width(), 0)

        srcPos = mp.Vector3(0, -cell.y / 2 + pmlLen + padding / 2)

        measurePos = mp.Vector3(
            0,
            cell.y / 2 - pmlLen - padding / 4,
        )

        sources: list[mp.Source] = [
            mp.Source(
                src=mp.GaussianSource(freq, fwidth=df),
                component=mp.Ez,  # type: ignore
                center=srcPos,
                size=src,
            )
        ]

        matMed = mp.Medium(index=self.design_region_ind)
        backMed = mp.Medium(index=self.sub_ind)

        matgrid_regions: list[mpa.DesignRegion] = []

        geometry = self.create_geometry()
        total_height = -cell.y / 2 + pmlLen + padding
        for i in range(self.n_design_regions):
            matGrid = mp.MaterialGrid(
                mp.Vector3(self.nx_design[i], self.ny_design[i]),
                backMed,
                matMed,
                weights=np.ones((self.nx_design[i], self.ny_design[i])) * 0.5,
                beta=0,
                do_averaging=optimization.use_epsavg,
                damping=optimization.use_damping,
                grid_type="U_MEAN",
            )

            matGridRegion = mpa.DesignRegion(
                matGrid,  # type: ignore
                volume=mp.Volume(
                    center=mp.Vector3(0, total_height + self.designY[i] / 2),
                    size=mp.Vector3(self.designX[i], self.designY[i]),
                ),
            )

            matgrid_regions.append(matGridRegion)

            total_height += self.designY[i] + self.design_region_spacing

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

        # We want to maximize the power going in to this mode:

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
        def J(far_field: List[List[complex]]) -> float:
            return -npa.abs(far_field[0, :, 2] ** 2)  # type: ignore

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=[J],
            objective_arguments=obList,
            design_regions=matgrid_regions,
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
        opt = self.create_sim(optimization)

        def obj(
            weights: npt.NDArray[np.float64],
        ) -> ah.ObjectiveReturn:

            val, grad = opt(self.separate_weights(weights))  # type: ignore
            print(f"val: {val}")
            print(f"max: {np.max(grad[:])}")

            # In case of a single design region, we need to convert it back into a
            # List[List[float]] as meep tries to be helpful by squeezing
            if not (
                isinstance(grad, list) and all(isinstance(item, list) for item in grad)
            ):
                grad = [grad]
            return (val, npa.concatenate(grad))  # type: ignore

        return obj

    def separate_weights(
        self, weights: npt.NDArray[np.float64]
    ) -> list[npt.NDArray[np.float64]]:
        ret: list[npt.NDArray[np.float64]] = []
        start = 0
        stop = 0
        for n in self.total_n():
            stop = start + n
            ret.append(weights[start:stop])
            start = stop

        return ret
