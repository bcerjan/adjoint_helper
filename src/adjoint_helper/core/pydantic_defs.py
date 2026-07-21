__all__ = ["FloatArray1D", "FloatArray2D", "BoolArray2D"]
import base64
from typing import Any, ClassVar, Dict
import numpy as np
from numpy.typing import NDArray
from pydantic_core import core_schema


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
                arr = v
            elif isinstance(v, dict) and "bytes" in v:
                raw_bytes = base64.b64decode(v["bytes"])
                arr = np.frombuffer(raw_bytes, dtype=np.dtype(v["dtype"]))
                arr = arr.reshape(v["shape"])
            else:
                try:
                    arr = np.array(v)
                except Exception as e:
                    raise ValueError(f"Could not convert input to numpy array: {e}")

            if arr.dtype != target_type:
                print(f"Warning: coercing type of input {arr.dtype} to {target_type}")
                arr = arr.astype(target_type)

            if cls._ndim is not None and arr.ndim != cls._ndim:
                raise ValueError(
                    f"Array must have exactly {cls._ndim} dimensions, got {arr.ndim}"
                )

            return arr

        def serialize(v: np.ndarray) -> Dict[str, Any]:
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
