import numpy as np

from adjoint_helper import SimulationReturn
from adjoint_helper.core.defs import (
    RawWeightsType,
    WeightsType,
    combine_objective_values,
)

type Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type MeepObjValReturn = float | list[float] | list[Array1D]
type MeepGradReturn = (
    list[list[Array2D]]
    | list[list[Array1D]]
    | list[Array2D]
    | list[Array1D]
    | Array2D
    | Array1D
)
type MeepOptReturn = tuple[MeepObjValReturn, MeepGradReturn]

objective_combination_vals = {" "}


def normalize_meep_weights(weights: list[WeightsType]) -> list[RawWeightsType]:
    """Converts the weights into the format that meep expects"""
    m_weights: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
    for w in weights:
        m_weights.append(w.flatten())

    return m_weights


def normalize_meep_output(
    opt_out: MeepOptReturn, operation: str = "mean"
) -> SimulationReturn:
    """Function to re-nest output from meep and re-organize it they way
    adjoint_helper expects. Meep tries to help by auto-squeezing results, but
    this breaks the internal handling of the return by adjoint_helper so we
    have to unsqueeze everything. We also need to re-order the results, as
    meep returns them as objective function -> design_region -> parameter/frequency
    (as a 2D array if there are more than one)

    Also will combine the objective values (if there is more than one parameter)
    using the combination method passed in. See `_OPERATION_REGISTRY` for the options.
    """

    obj_val = opt_out[0]
    obj_grad = opt_out[1]
    if isinstance(obj_val, list) and isinstance(obj_val[0], np.ndarray):
        obj_val = combine_objective_values(obj_val, operation)  # type: ignore
    if isinstance(obj_val, float):
        obj_val = [obj_val]

    def get_list_depth(lst):
        if isinstance(lst, np.ndarray):
            return 0
        if isinstance(lst, list) and len(lst) > 0:
            return 1 + get_list_depth(lst[0])
        return 1

    current_depth = get_list_depth(obj_grad)
    if current_depth >= 4:
        raise ValueError("too much nesting in normalize_meep_output")

    while current_depth < 3:
        normalized_grad = [obj_grad]
        current_depth += 1

    normalized_grad = np.array(obj_grad)
    normalized_grad = np.transpose(normalized_grad, (0, 1, 3, 2))

    ret = (obj_val, normalized_grad)

    return ret  # type: ignore
