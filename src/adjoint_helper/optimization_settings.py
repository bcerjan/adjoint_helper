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

import numpy as np
import numpy.typing as npt
from vendors.meep.filters import get_conic_radius_from_eta_e  # type: ignore
from simulation_settings import SimulationSettings
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # type: ignore
from pathlib import Path
from abc import ABC, abstractmethod


class OptimizationSettings(ABC):
    """
    Class to store necessary optimization parameters as well as optimization
    history.
    """

    obj: list[np.float64]
    data: list[np.float64]
    weights: list[npt.NDArray[np.float64]]
    grad: list[np.float64]
    connectivity: list[int]
    do_connectivity: bool
    sigmoid_threshold: float
    sigmoid_erosion: float
    sigmoid_dilation: float
    sigmoid_biases: list[float]
    sigmoid_bias: float
    sigmoid_bias_sim: float
    sigmoid_bias_threshold: float
    epigraph_tolerance: float
    connectivity_sigmoid_threshold: float
    _apply_connectivity: bool
    do_connectivity: bool
    linewidth_sigmoid_threshold: float
    _apply_linewidth: bool
    max_evals: list[int]
    filter_radius: float
    use_epsavg: bool = False
    use_damping: bool = False
    maximum_runtime: float
    minimum_runtime: float
    decay_by: float
    use_smoothed_projection: bool
    last_completed_index: int

    def __init__(
        self,
        minimum_size: float = 0.05,
        sigmoid_bias_threshold: float = 32,  # Sigmoid bias at which eps_avg turns on
        sigmoid_threshold: float = 0.5,  # Eta
        sigmoid_erosion: float = 0.65,  # Eta_e
        sigmoid_biases: list[float] = [4, 8, 16, 24, 32, 40],
        connectivity_sigmoid_threshold: float = 16,
        linewidth_sigmoid_threshold: float = 24,  # Sigmoid bias at which line width constraint turns on
        max_evals: list[int] | int = 10,  # if int, all biases get same number
        maximum_runtime: float = 200,
        minimum_runtime: float = 0,
        decay_by: float = 1e-6,
        use_smoothed_projection: bool = False,
        do_connectivity: bool = False,
    ):
        """"""
        evals: list[int] = []

        if type(max_evals) is list and len(sigmoid_biases) != len(max_evals):
            raise ValueError("Mismatch between length of sigmoid_biases and max_evals")

        if type(max_evals) is int:
            evals = [max_evals for _ in sigmoid_biases]
        else:
            evals = max_evals  # type: ignore

        self.obj = []
        self.data = []
        self.weights = []
        self.grad = []
        self.connectivity = []
        self.last_completed_index = -1

        self.filter_radius = get_conic_radius_from_eta_e(  # type: ignore
            minimum_size, sigmoid_erosion
        )

        self.sigmoid_bias = sigmoid_biases[0]
        self.sigmoid_biases = sigmoid_biases
        self.sigmoid_bias_threshold = sigmoid_bias_threshold
        self.sigmoid_threshold = sigmoid_threshold
        self.sigmoid_erosion = sigmoid_erosion
        self.sigmoid_dilation = 1 - sigmoid_erosion
        self.max_evals = evals
        self.maximum_runtime = maximum_runtime
        self.minimum_runtime = minimum_runtime
        self.decay_by = decay_by
        self.apply_connectivity = False
        self.apply_linewidth = False
        self.do_connectivity = do_connectivity
        self.connectivity_sigmoid_threshold = connectivity_sigmoid_threshold
        self.linewidth_sigmoid_threshold = linewidth_sigmoid_threshold
        self.use_smoothed_projection = use_smoothed_projection

    # Change this to a bunch of getters rather than this kind of function.
    # This is doing unnecessary stuff.
    def apply_settings(self, current_sigmoid_bias: float) -> None:
        """Updates the optimization object to tell it to use/not use eps_avg,
        damping, connectivity constraint, and linewidth constraint. Also updates
        the internally stored `sigmoid_bias`

        Args:
            current_sigmoid_bias (float): Current bias of the optimization
        """
        self.sigmoid_bias = current_sigmoid_bias

        if current_sigmoid_bias >= self.sigmoid_bias_threshold:
            self.use_epsavg = True
            # eps_avg and damping are (somewhat) mutually exclusive, see:
            # https://github.com/NanoComp/photonics-opt-testbed/issues/31#issuecomment-1370041394
            self.use_damping = False
        else:
            self.use_epsavg = False
            self.use_damping = True

        if (
            current_sigmoid_bias > self.connectivity_sigmoid_threshold
            and self.do_connectivity
        ):
            self.apply_connectivity = True
        else:
            self.apply_connectivity = False

        if current_sigmoid_bias > self.linewidth_sigmoid_threshold:
            self.apply_linewidth = True
        else:
            self.apply_linewidth = False

    @abstractmethod
    def optimize(self, settings: SimulationSettings) -> npt.NDArray[np.float64]:
        pass

    @property
    def bias(self) -> float:
        return self.sigmoid_biases[self.last_completed_index + 1]

    @property
    def apply_linewidth(self) -> bool:
        return (self.bias > self.linewidth_sigmoid_threshold) or self._apply_linewidth

    @apply_linewidth.setter
    def apply_linewidth(self, val: bool):
        self._apply_linewidth = val

    @property
    def apply_connectivity(self) -> bool:
        return (
            (self.bias > self.connectivity_sigmoid_threshold) and self.do_connectivity
        ) or self._apply_connectivity

    @apply_connectivity.setter
    def apply_connectivity(self, val: bool):
        self._apply_linewidth = val


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

    temp_directory: Path
    k: float  # sigma for Gaussian filter for dataset generation
    datasize: int  # for dataset generation, how many training images should
    # be generated?

    def __init__(
        self,
        k: float = 4,
        datasize: int = 30_000,
        temp_directory: str | Path = "adjoint_tmp/",
        minimum_size: float = 0.05,
        sigmoid_bias_threshold: float = 32,  # Sigmoid bias at which eps_avg turns on
        sigmoid_threshold: float = 0.5,  # Eta
        sigmoid_erosion: float = 0.65,  # Eta_e
        sigmoid_biases: list[float] = [4, 8, 16, 24, 32, 40],
        connectivity_sigmoid_threshold: float = 16,
        line_width_sigmoid_threshold: float = 24,  # Sigmoid bias at which line width constraint turns on
        max_evals: list[int] | int = 10,  # if int, all biases get same number
        maximum_runtime: float = 200,
        minimum_runtime: float = 0,
        decay_by: float = 1e-6,
        use_smoothed_projection: bool = False,
        do_connectivity: bool = False,
    ):
        self.k = k
        self.datasize = datasize
        self.temp_directory = Path(temp_directory).resolve()

        super().__init__(
            minimum_size=minimum_size,
            sigmoid_bias_threshold=sigmoid_bias_threshold,
            sigmoid_threshold=sigmoid_threshold,
            sigmoid_erosion=sigmoid_erosion,
            sigmoid_biases=sigmoid_biases,
            connectivity_sigmoid_threshold=connectivity_sigmoid_threshold,
            linewidth_sigmoid_threshold=line_width_sigmoid_threshold,
            max_evals=max_evals,
            maximum_runtime=maximum_runtime,
            minimum_runtime=minimum_runtime,
            decay_by=decay_by,
            use_smoothed_projection=use_smoothed_projection,
            do_connectivity=do_connectivity,
        )

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
