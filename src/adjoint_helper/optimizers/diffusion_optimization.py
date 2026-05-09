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

from pathlib import Path
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from ..core.base_settings import OptimizationSettings
from ..vendors.meep.filters import gaussian_filter  # type: ignore


class AdjointDiffusionSettings(OptimizationSettings):
    """Subclass of `OptimizationSettings`, specialized to work with the
    `AdjointDiffusion` optimization backend. By default will create a temporary
    directory structure in "./adjoint_tmp/", but this can be specialized if
    desired. A trained model will live in that directory (if you do training)
    and can be re-loaded to use for other optimizations.

    Incorporates modified versions of AdjointDiffusion's 'dataset_generation.py'
    code. "image" functions in this class work with uint8 values (so 0-255) rather
    than 0-1 as "ordinary" weights would be.

    For AdjointDiffusion, your design region must be a square, with (ideally) sides
    that are multiple of 64 (128, 256, or 512). If your region is close to that,
    it will be padded/shrunk. If it is not, it will be treated as many overlapping
    sub-regions, but this is substantially less efficient.
    """

    temp_directory: Path = Path("./adjoint_tmp").resolve()
    k: float = 4  # sigma for Gaussian filter for dataset generation
    datasize: int = 30_000  # for dataset generation, how many training images should
    # be generated?

    # def __init__(
    #     self,
    #     k: float = 4,
    #     datasize: int = 30_000,
    #     temp_directory: str | Path = "adjoint_tmp/",
    #     minimum_size: float = 0.05,
    #     sigmoid_bias_threshold: float = 32,  # Sigmoid bias at which eps_avg turns on
    #     sigmoid_threshold: float = 0.5,  # Eta
    #     sigmoid_erosion: float = 0.65,  # Eta_e
    #     sigmoid_biases: list[float] = [4, 8, 16, 24, 32, 40],
    #     connectivity_sigmoid_threshold: float = 16,
    #     line_width_sigmoid_threshold: float = 24,  # Sigmoid bias at which line width constraint turns on
    #     max_evals: list[int] | int = 10,  # if int, all biases get same number
    #     maximum_runtime: float = 200,
    #     minimum_runtime: float = 0,
    #     decay_by: float = 1e-6,
    #     use_smoothed_projection: bool = False,
    #     do_connectivity: bool = False,
    # ):
    #     self.k = k
    #     self.datasize = datasize
    #     self.temp_directory = Path(temp_directory).resolve()

    #     super().__init__(
    #         minimum_size=minimum_size,
    #         sigmoid_bias_threshold=sigmoid_bias_threshold,
    #         sigmoid_threshold=sigmoid_threshold,
    #         sigmoid_erosion=sigmoid_erosion,
    #         sigmoid_biases=sigmoid_biases,
    #         connectivity_sigmoid_threshold=connectivity_sigmoid_threshold,
    #         linewidth_sigmoid_threshold=line_width_sigmoid_threshold,
    #         max_evals=max_evals,
    #         maximum_runtime=maximum_runtime,
    #         minimum_runtime=minimum_runtime,
    #         decay_by=decay_by,
    #         use_smoothed_projection=use_smoothed_projection,
    #         do_connectivity=do_connectivity,
    #     )

    def initialize_directories(self) -> None:
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        (self.temp_directory / f"sigma{self.k}/struct").mkdir(exist_ok=True)
        (self.temp_directory / "models").mkdir(exist_ok=True)

    def generate_random_binary_image(
        self,
        size: int,
    ) -> npt.NDArray[np.uint8]:
        """Generate a random binary image for each pixel in the design region"""
        return np.random.randint(0, 2, (size, size), dtype=np.uint8) * 255

    def apply_gaussian_filter(
        self, image: npt.NDArray[np.uint8], sigma: float
    ) -> npt.NDArray[np.float64]:
        """Apply a Gaussian filter to a binary image."""
        return gaussian_filter(image, sigma=sigma)  # type: ignore

    def binarize_image(
        self, image: npt.NDArray[np.float64], threshold: float = 255 / 2
    ) -> npt.NDArray[np.int_]:
        """Binarize an image based on a threshold."""
        return (image > threshold).astype(int) * 255

    def save_image_as_png(
        self,
        image: npt.NDArray[np.uint8],
        fname: str = "image.png",
    ):
        """Save the image as a PNG."""
        fpath = self.temp_directory / f"sigma{self.k}/struct" / fname
        plt.imsave(fpath, image, cmap="gray")  # type: ignore
