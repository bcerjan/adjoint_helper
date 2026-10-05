__all__ = ["BoolArray2D", "FloatArray1D", "FloatArray2D"]
import base64
import logging
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from pydantic_core import core_schema

logger = logging.getLogger(__name__)


class PydanticNDArray(np.ndarray):
    _dtype: ClassVar[Any] = np.float64
    _ndim: ClassVar[int | None] = None

    def __new__(cls, *args, **kwargs) -> NDArray[Any]:
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        target_type = np.dtype(cls._dtype)

        def validate(v: Any) -> np.ndarray:

            if isinstance(v, np.ndarray):
                arr = v.copy()
            elif isinstance(v, dict) and "bytes" in v:
                raw_bytes = base64.b64decode(v["bytes"])
                arr = np.frombuffer(raw_bytes, dtype=np.dtype(v["dtype"]))
                arr = arr.reshape(v["shape"])
                arr = arr.copy()
            else:
                try:
                    arr = np.array(v)
                except Exception as e:
                    raise ValueError(f"Could not convert input to numpy array: {e}")

            if arr.dtype != target_type:
                logger.warning(
                    f"Warning: coercing type of input {arr.dtype} to {target_type}"
                )
                arr = arr.astype(target_type)

            if cls._ndim is not None and arr.ndim != cls._ndim:
                raise ValueError(
                    f"Array must have exactly {cls._ndim} dimensions, got {arr.ndim}"
                )

            return arr

        def serialize(v: np.ndarray) -> dict[str, Any]:
            raw_bytes = base64.b64encode(v.tobytes()).decode("utf-8")
            return {"shape": list(v.shape), "dtype": str(v.dtype), "bytes": raw_bytes}

        serialization_scheme = core_schema.plain_serializer_function_ser_schema(
            serialize
        )

        return core_schema.no_info_plain_validator_function(
            validate, serialization=serialization_scheme
        )


class FloatArray1D(PydanticNDArray):
    _dtype = np.float64
    _ndim = 1


class FloatArray2D(PydanticNDArray):
    _dtype = np.float64
    _ndim = 2


class BoolArray2D(PydanticNDArray):
    _dtype = np.bool
    _ndim = 2


class IntArray2D(PydanticNDArray):
    _dtype = int
    _ndim = 2


# TODO: Make it so we can store the solver / configuraton / state
# for people. Right now there isn't a particularly good way to do this
# without requiring substantial setup on my end to define all of the
# default parameters for any optimizer anyone might want to use.
# The way we would handle this is when creating an `OptaxOptimization` we
# would also require that the `params` dict be passed with any custom
# parameter changes (and possibly with the algorithm name?) so that it
# could be recreated later.
"""
Allowed libraries are: nlopt, optax

For nlopt, one of the params _must_ be "n_parameters"
as that is required to re-create the optimizer

For optax, see `ALGO_MAP` below for a list of supported algorithms
for save/load (at this time). If you want another one added,
just submit a PR that adds it there (along with any required parameter
checks needed for that algorithm)
"""


class SolverConfig(BaseModel):
    library: str
    algorithm: str
    params: dict[str, Any] = Field(default_factory=dict)


class SolverFactory:
    @staticmethod
    def create(config: SolverConfig):
        if config.library == "nlopt":
            import nlopt

            algo = getattr(nlopt, config.algorithm)
            params = config.params.pop("n_parameters")
            solver = nlopt.opt(algo, params)

            for k, v in config.params.items():
                solver.set_param(k, v)

            return solver

        if config.library == "optax":
            return SolverFactory._build_optax(config.algorithm, config.params)

    @staticmethod
    def _build_optax(algo_name: str, params: dict[str, Any]):
        import optax

        ALGO_MAP = {
            "adam": optax.adam,
            "sgd": optax.sgd,
            "rmsprop": optax.rmsprop,
            "adagrad": optax.adagrad,
            "adamw": optax.adamw,
        }

        if algo_name.lower() not in ALGO_MAP:
            raise ValueError(
                f"Optax algorithm '{algo_name}' not supported for loading. "
                f"Available options are: {list(ALGO_MAP.keys())}"
            )

        optimizer_fn = ALGO_MAP[algo_name.lower()]

        try:
            return optimizer_fn(**params)
        except TypeError as e:
            raise ValueError(f"Invalid parameters for {algo_name}: {e}")
