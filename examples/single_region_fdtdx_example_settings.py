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


import logging
from time import time
from typing import Any, Self

import fdtdx
import jax
import jax.numpy as jnp
import matplotlib
import pytreeclass as tc
from fdtdx import colors

matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import numpy as np
from pydantic import computed_field, model_validator

import adjoint_helper as ah
from adjoint_helper.utils.normalization_functions import (
    normalize_fdtdx_output,
    normalize_fdtdx_weights,
)

"""
Example of a settings class -- in theory the only piece of code you need to write.
This example optimizes a waveguide pointed in the left direction with a variable
sized design region.

Note that the creation of the simulation is separate from creation of the objective.
This isn't necessary, but it makes it easier to do visualization stuff as you
can get the bare simulation instead of the objective (which obscures it)

This example is using FDTDx as the simulation backend. Note that because it uses
JAX directly, setting up a multi-optimization problem is a little more complicated.

Also note that on my test computer, the `arrays` parameter is too big to fit on
my GPU so I had to pin it to the CPU device instead. This is done at "# CPU PIN" if
you want to search for it.
"""

logger = logging.getLogger(__name__)


class SingleRegionExampleSettings(ah.SimulationSettings):
    """These are parameters that are unique to your simulation and can be re-used
    for different variations (new substrate / materials / whatever)

    For FDTDx, resolution defines the number of points per `units` distance

    All distances will be multiplied by `units` before being passed as "physical"
    values to FDTDx
    """

    beamSize: float
    beamTilt: float  # In degrees, positive
    wgWidth: float
    wgHeight: float
    matInd: float
    subInd: float
    bgInd: float
    wavelength: float
    resolution: int
    units: float = 1e-6
    seed: int = 442

    _config: fdtdx.SimulationConfig | None = None
    _object_list: list = []
    _needs_jit: bool = True
    _placement_return = None

    @property
    def _pml_len(self) -> float:
        return self.wavelength * 2 * self.units

    @property
    def _padding(self) -> float:
        return self.wavelength * 2 * self.units

    @property
    def _design_x(self) -> float:
        return self.design_regions[0].x_size * self.units

    @property
    def _design_y(self) -> float:
        return self.design_regions[0].y_size * self.units

    @property
    def _wg_height(self) -> float:
        return self.wgHeight * self.units

    @property
    def _wg_width(self) -> float:
        return self.wgWidth * self.units

    @computed_field
    @property
    def _period(self) -> float:
        return fdtdx.constants.wavelength_to_period(self.wavelength * self.units)

    @property
    def _df(self) -> fdtdx.WaveCharacter:
        sigma_t = 4.0e-15  # TODO: What is going on here...
        return fdtdx.WaveCharacter(frequency=1.0 / (2.0 * np.pi * sigma_t))

    @property
    def _material_config(self) -> dict:
        return {
            "Background": fdtdx.Material(permittivity=self.bgInd**2),
            "Device_Material": fdtdx.Material(permittivity=self.matInd**2),
        }

    @property
    def _voxel_shape(self) -> tuple[float, float, float]:
        vox = self.units / self.resolution
        return (vox, vox, vox)

    @model_validator(mode="after")
    def set_differentiation(self) -> Self:
        ah.anp.set_differentiation_lib("jax")
        return self

    @model_validator(mode="after")
    def initialize_config(self) -> Self:
        spacing = self.units / self.resolution
        self._config = fdtdx.SimulationConfig(
            time=self.maximum_runtime,
            resolution=spacing,
            dtype=jnp.float32,
            courant_factor=0.99,
        )

        # if self.enforce_symmetry:
        #     self._config.at["symmetry"].set((0, -1, 0))
        # self._config.aset("symmetry", (0, -1, 0))

        grad_conf = fdtdx.GradientConfig(
            recorder=fdtdx.Recorder(
                modules=[
                    fdtdx.DtypeConversion(dtype=jnp.float8_e4m3fnuz),
                    fdtdx.LinearReconstructEveryK(k=5),
                ]
            )
        )

        self._config.aset("gradient_config", grad_conf)
        return self

    @property
    def _volume(self) -> fdtdx.SimulationVolume:
        return fdtdx.SimulationVolume(
            partial_real_shape=(
                self._design_x + 2 * self._pml_len + 2 * self._padding,
                self._design_y + 2 * self._pml_len + 2 * self._padding,
                self._wg_height + 2 * self._padding,
            ),
            name="volume",
        )

    @property
    def _boundary_config(self) -> fdtdx.BoundaryConfig:
        thickness = int(self._pml_len / self.units)
        return fdtdx.BoundaryConfig.from_uniform_bound(thickness=thickness)

    @property
    def _boundary_objects(
        self,
    ) -> tuple[dict, list[fdtdx.PositionConstraint]]:
        return fdtdx.boundary_objects_from_config(self._boundary_config, self._volume)

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

        if self.enforce_symmetry:
            a = 1  # TODO:

        return 1.0

    def create_sim(
        self, optimization: ah.OptimizationSettings
    ) -> tuple[
        fdtdx.ObjectContainer,
        fdtdx.ArrayContainer,
        fdtdx.ParameterContainer,
        fdtdx.SimulationConfig,
        dict[str, Any],
    ]:
        """
        Creates an mpa Optimization Problem for optimizing a grating coupler mated to
        a waveguide. Works assuming the waveguide is in the x-y plane and exits the
        simulation in the -x direction (left).

        Args:
        optimization: OptimizationSettings containing optimization parameters

        """

        if self._placement_return is None:
            volume = self._volume
            self._object_list = [volume]
            bndry_dict, c_list = self._boundary_objects
            self._object_list.extend(list(bndry_dict.values()))
            placement_constraints = c_list

            substrate = fdtdx.UniformMaterialObject(
                partial_real_shape=(None, None, self._padding),
                material=fdtdx.Material(permittivity=self.subInd**2),
                # color=fdtdx.colors.XKCD_ORANGE,
                name="substrate",
            )

            placement_constraints.append(
                substrate.place_relative_to(
                    volume, axes=2, own_positions=-1, other_positions=-1
                )
            )

            self._object_list.append(substrate)

            # IMPORTANT!
            # If you use the built-in normalization functions, it is assumed
            # that your devices are named like below (`design_regionXXX`)
            # If you call them something else, you need to handle the weights
            # conversion yourself
            device = fdtdx.Device(
                name="design_region0",
                partial_real_shape=(
                    self._design_x,
                    self._design_y,
                    self._wg_height,
                ),
                materials=self._material_config,
                partial_voxel_real_shape=(
                    self._voxel_shape[0],
                    self._voxel_shape[1],
                    self._wg_height,
                ),
                # Adjoint Helper does these internally, so don't apply them
                # here as well
                param_transforms=[],
            )

            placement_constraints.append(device.place_above(substrate))
            placement_constraints.append(device.place_at_center(substrate, axes=(0, 1)))

            self._object_list.append(device)

            wg = fdtdx.UniformMaterialObject(
                partial_real_shape=(
                    None,
                    self._wg_width,
                    self._wg_height,
                ),
                material=self._material_config["Device_Material"],
                name="wg",
                color=colors.CYAN,
            )

            placement_constraints.extend(
                [
                    wg.place_at_center(device, axes=1),
                    wg.extend_to(device, axis=0, direction="+"),
                    wg.place_above(substrate),
                ]
            )

            self._object_list.append(wg)

            detector = fdtdx.ModeOverlapDetector(
                mode_index=0,
                scaling_mode="pulse",
                filter_pol="te",
                partial_grid_shape=(1, None, None),
                partial_real_shape=(None, 6 * self._wg_width, 6 * self._wg_height),
                direction="-",
                wave_characters=[
                    fdtdx.WaveCharacter(wavelength=self.wavelength * self.units),
                ],
                name="mode_detector",
            )

            placement_constraints.extend(
                [
                    detector.place_at_center(device, axes=(1, 2)),
                    detector.place_relative_to(
                        wg,
                        axes=0,
                        own_positions=1,
                        other_positions=-1,
                        grid_margins=-self._boundary_config.thickness_grid_minx + 5,
                    ),
                ]
            )

            self._object_list.append(detector)
            center_wave = fdtdx.WaveCharacter(wavelength=self.wavelength * self.units)

            source = fdtdx.GaussianPlaneSource(
                wave_character=center_wave,
                fixed_E_polarization_vector=(0, 1, 0),
                radius=self.beamSize,
                direction="-",
                elevation_angle=ah.anp.pi / 180 * -self.beamTilt,
                partial_grid_shape=(None, None, 1),
                partial_real_shape=(
                    self.beamSize * self.units * 3,
                    self.beamSize * self.units * 3,
                    None,
                ),
                temporal_profile=fdtdx.GaussianPulseProfile(
                    spectral_width=self._df, center_wave=center_wave
                ),
                name="source",
            )

            placement_constraints.extend(
                [
                    source.place_at_center(device, axes=(0, 1)),
                    source.place_relative_to(
                        volume,
                        axes=2,
                        own_positions=1,
                        other_positions=1,
                        grid_margins=self._boundary_config.thickness_grid_maxz - 5,
                    ),
                ]
            )

            self._object_list.append(source)

            key, subkey = jax.random.split(jax.random.PRNGKey(self.seed))
            objects, arrays, params, config, info = fdtdx.place_objects(
                object_list=self._object_list,
                config=self._config,
                constraints=placement_constraints,
                key=subkey,
            )

            fdtdx.apply_params(
                arrays=arrays, objects=objects, params=params, key=key, beta=4
            )

            self._placement_return = (objects, arrays, params, config, info)

        return self._placement_return

    def plot_sim(self, optimization: ah.OptimizationSettings) -> None:
        objects, arrays, params, config, _ = self.create_sim(optimization=optimization)
        fig = fdtdx.plot_setup(config=config, objects=objects)
        plt.show()

    def print_usage(self, optimization: ah.OptimizationSettings) -> None:
        objects, arrays, params, config, _ = self.create_sim(optimization=optimization)
        print(tc.tree_summary(arrays, depth=1))

    def create_opt(
        self, optimization: ah.OptimizationSettings
    ) -> ah.OptimizationFunction:

        objects, arrays, params, config, _ = self.create_sim(optimization=optimization)

        sigmoid_bias = ah.anp.array(optimization.sigmoid_bias)
        normalization_value = ah.anp.array(self.baseline_optimization_value)

        # TODO: It feels like there should be a better way?
        # CPU PIN
        # cpu_device = jax.devices("cpu")[0]
        # bound_data = jax.device_put(arrays, cpu_device)
        bound_data = arrays

        #
        # Due to how things with JAX work, this needs to be inside a closure
        # or it can't JIT it appropriately.
        def loss_fn(
            params: fdtdx.ParameterContainer,
            arrays: fdtdx.ArrayContainer,
            key: jax.Array,
        ):
            arrays_updated, objects_updated, _ = fdtdx.apply_params(
                arrays=arrays,
                objects=objects,
                params=params,
                key=key,
                beta=sigmoid_bias,
            )

            final_state = fdtdx.run_fdtd(
                arrays=arrays_updated,
                objects=objects_updated,
                config=self._config,
                key=key,
            )

            _, arrays = final_state

            mode_flux = arrays.detector_states["mode_detector"]["phasor"].sum()
            mode_flux = ah.anp.abs(mode_flux)
            objective = (
                1 - mode_flux / normalization_value
            )  # 1 minus to invert for minimization

            return objective

        key, subkey = jax.random.split(jax.random.PRNGKey(self.seed - 18))

        if self._needs_jit:
            logger.info("Starting JIT compilation -- this can take a while")
            start_time = time()

            self._jitted_loss = (
                jax.jit(
                    jax.value_and_grad(loss_fn),
                    donate_argnames=["arrays"],
                )
                .lower(params, bound_data, key)
                .compile()
            )

            self._needs_jit = False

            jit_time = time() - start_time
            logger.info(f"Compilationg succeeded in {jit_time} seconds")

        def obj(
            weights: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
        ) -> ah.SimulationReturn:

            fdtdx_weights = normalize_fdtdx_weights(weights=weights)
            # fdtdx_weights = ParameterContainer(fdtdx_weights)

            opt_out = self._jitted_loss(
                fdtdx_weights,
                bound_data,
                subkey,
            )

            # For common backends, there should be a similar utility function
            # to automatically normalize the output into the correct shape
            # for the rest of the library. If there isn't one for your backend,
            # feel free to write one and submit it as a PR.
            val, grad = normalize_fdtdx_output(opt_out=opt_out, settings=self)

            return (val, grad)

        return obj
