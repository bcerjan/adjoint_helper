import numpy as np
from adjoint_helper import SimulationReturn
from adjoint_helper.core.defs import WeightsType, RawWeightsType


type Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type meep_obj_val_return = float | list[float] | list[Array1D]
type meep_grad_return = (
    list[list[Array2D]]
    | list[list[Array1D]]
    | list[Array2D]
    | list[Array1D]
    | Array2D
    | Array1D
)
type meep_opt_return = tuple[meep_obj_val_return, meep_grad_return]


def normalize_meep_weights(weights: list[WeightsType]) -> list[RawWeightsType]:
    m_weights: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
    for w in weights:
        m_weights.append(w.flatten())

    return m_weights


def normalize_meep_output(opt_out: meep_opt_return) -> SimulationReturn:
    """Function to re-nest output from meep and re-organize it they way
    adjoint_helper expects. Meep tries to help by auto-squeezing results, but
    this breaks the internal handling of the return by adjoint_helper so we
    have to unsqueeze everything. We also need to re-order the results, as
    meep returns them as objective function -> design_region -> parameter/frequency
    (as a 2D array if there are more than one)"""

    obj_val = opt_out[0]
    obj_grad = opt_out[1]
    if isinstance(obj_val, np.ndarray):
        obj_val = obj_val.tolist()
    if not isinstance(obj_val, list):
        obj_val = [float(obj_val)]

    def separate_freqs(item):
        if isinstance(item, np.ndarray):
            if item.ndim == 2:
                return [col for col in item]
            elif item.ndim == 1:
                return [item]
            raise ValueError(f"Unsupported array dimension: {item.ndim}")

        return [separate_freqs(sub_item) for sub_item in item]

    normalized_grad = separate_freqs(obj_grad)

    def get_list_depth(lst):
        if isinstance(lst, np.ndarray):
            return 0
        if isinstance(lst, list) and len(lst) > 0:
            return 1 + get_list_depth(lst[0])
        return 1

    current_depth = get_list_depth(normalized_grad)
    if current_depth >= 4:
        raise ValueError("too much nesting in correct_opt_output")

    while current_depth < 3:
        normalized_grad = [normalized_grad]
        current_depth += 1

    # Now we need to re-organize into the correct order:
    normalized_grad = [list(layer) for layer in zip(*normalized_grad)]

    ret = (obj_val, normalized_grad)

    return ret  # type: ignore
