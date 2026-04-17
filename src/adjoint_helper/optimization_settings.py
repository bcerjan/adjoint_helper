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
import meep.adjoint as mpa  # type: ignore


class OptimizationSettings:
    """
    Class to store necessary optimization parameters as well as optimization
    history.
    """

    obj: list[np.float_]
    data: list[np.float_]
    weights: list[npt.NDArray[np.float_]]
    grad: list[np.float_]
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
    connectivity_tolerance: float
    connectivity_sigmoid_threshold: float
    apply_connectivity: bool
    do_connectivity: bool
    line_width_sigmoid_threshold: float
    apply_linewidth: bool
    max_evals: list[int]
    filter_radius: float
    use_epsavg: bool = False
    use_damping: bool = False
    maximum_runtime: float
    minimum_runtime: float
    decay_by: float
    use_smoothed_projection: bool
    last_completed_index: int

    nlopt_line_width_tol: float
    penalty_weight: float

    def __init__(
        self,
        minimum_size: float = 0.05,
        sigmoid_bias_threshold: float = 32,  # Sigmoid bias at which eps_avg turns on
        sigmoid_threshold: float = 0.5,  # Eta
        sigmoid_erosion: float = 0.65,  # Eta_e
        sigmoid_biases: list[float] = [4, 8, 16, 24, 32, 40],
        connectivity_tolerance: float = 1e-3,
        connectivity_sigmoid_threshold: float = 16,
        line_width_sigmoid_threshold: float = 24,  # Sigmoid bias at which line width constraint turns on
        max_evals: list[int] | int = 10,  # if int, all biases get same number
        maximum_runtime: float = 200,
        minimum_runtime: float = 0,
        decay_by: float = 1e-6,
        use_smoothed_projection: bool = False,
        do_connectivity: bool = False,
        nlopt_line_width_tol: float = 1e-4,  # Tolerance for nlopt constraint
        penalty_weight: float = 0.1,  # only used for non-nlopt
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

        self.filter_radius = mpa.get_conic_radius_from_eta_e(  # type: ignore
            minimum_size, sigmoid_erosion
        )

        self.sigmoid_bias = sigmoid_biases[0]
        self.sigmoid_biases = sigmoid_biases
        self.sigmoid_bias_threshold = sigmoid_bias_threshold
        self.sigmoid_threshold = sigmoid_threshold
        self.sigmoid_erosion = sigmoid_erosion
        self.sigmoid_dilation = 1 - sigmoid_erosion
        self.connectivity_tolerance = connectivity_tolerance
        self.nlopt_line_width_tol = nlopt_line_width_tol
        self.max_evals = evals
        self.maximum_runtime = maximum_runtime
        self.minimum_runtime = minimum_runtime
        self.decay_by = decay_by
        self.apply_connectivity = False
        self.apply_linewidth = False
        self.do_connectivity = do_connectivity
        self.connectivity_sigmoid_threshold = connectivity_sigmoid_threshold
        self.line_width_sigmoid_threshold = line_width_sigmoid_threshold
        self.penalty_weight = penalty_weight
        self.use_smoothed_projection = use_smoothed_projection

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

        if current_sigmoid_bias > self.line_width_sigmoid_threshold:
            self.apply_linewidth = True
        else:
            self.apply_linewidth = False
