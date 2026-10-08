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

__all__ = ["anp"]

from copy import copy
from types import ModuleType
from typing import TYPE_CHECKING, Any
from weakref import proxy

import numpy as real_np

AVAILABLE_DIFF_LIBS = {"autograd", "jax"}


class AtIndexer:
    arr: real_np.ndarray

    def __init__(self, arr):
        self.arr = arr
        self.proxy = proxy

    def __getitem__(self, index):
        return AtSetter(self.arr, index)


class AtSetter:
    arr: real_np.ndarray
    index: int

    def __init__(self, arr, index) -> None:
        self.arr = arr
        self.index = index

    def set(self, value):
        arr_copy = copy(self.arr)
        arr_copy[self.index] = value
        return arr_copy

    def add(self, value):
        arr_copy = copy(self.arr)
        arr_copy[self.index] += value
        return arr_copy


class DifferentiationProxy:
    _active_diff_lib: str

    def __init__(self) -> None:
        self._active_diff_lib = "autograd"

    def set_differentiation_lib(self, lib: str) -> None:
        if lib not in AVAILABLE_DIFF_LIBS:
            raise ValueError(f"Differentiation Lib must be in {AVAILABLE_DIFF_LIBS}")
        self._active_diff_lib = lib

    @property
    def backend(self) -> str:
        return self._active_diff_lib

    @property
    def _numpy(self) -> ModuleType:
        if self._active_diff_lib == "jax":
            import jax.numpy as jnp

            return jnp
        if self._active_diff_lib == "autograd":
            import autograd.numpy as npa

            return npa

        raise ValueError("Invalid diff. library")

    @property
    def float_type(self):
        if self._active_diff_lib == "jax":
            from jax import numpy as jnp

            return jnp.float32

        if self._active_diff_lib == "autograd":
            from autograd import numpy as npa

            return npa.float64

    # custom `.at` so we can share syntax between jax and autograd
    def at(self, arr: Any) -> Any:
        if self.backend == "jax":
            return arr.at
        if self.backend == "autograd":
            return AtIndexer(arr)

    # Override Functions:
    def grad(self, fun, argnum=0, *args, **kwargs):
        if self._active_diff_lib == "jax":
            import jax

            return jax.grad(fun, argnum, *args, **kwargs)

        if self._active_diff_lib == "autograd":
            import autograd

            return autograd.grad(fun, argnum, *args, **kwargs)

    def value_and_grad(self, fun, argnum=0, *args, **kwargs):
        if self._active_diff_lib == "jax":
            import jax

            return jax.value_and_grad(fun, argnum, *args, **kwargs)

        if self._active_diff_lib == "autograd":
            import autograd

            return autograd.value_and_grad(fun, argnum, *args, **kwargs)

    def jacobian(self, fun, argnum=0, *args, **kwargs):
        if self._active_diff_lib == "jax":
            import jax

            return jax.jacrev(fun, argnum, *args, **kwargs)

        if self._active_diff_lib == "autograd":
            import autograd

            return autograd.jacobian(fun, argnum, *args, **kwargs)

    def tensor_jacobian_product(self, fun, argnum=0, *args, **kwargs):
        if self._active_diff_lib == "jax":
            import jax

            def tjp_fun(*args, **kwargs):
                primals, vector = args[:-1], args[-1]

                def target_fun(x):
                    new_args = list(primals)
                    new_args[argnum] = x
                    return fun(*new_args, **kwargs)

                target_primal = primals[argnum]
                _, vjp_fun = jax.vjp(target_fun, target_primal)
                return vjp_fun(vector)[0]

            return tjp_fun

        if self._active_diff_lib == "autograd":
            import autograd

            return autograd.differential_operators.tensor_jacobian_product(
                fun, argnum, *args, **kwargs
            )

    def __getattr__(self, name: str) -> Any:
        bypass_list = {
            "dtype",
            "generic",
            "number",
            "integer",
            "floating",
            "complexfloating",
            "float64",
            "float32",
            "float16",
            "double",
            "single",
            "int64",
            "int32",
            "int16",
            "int8",
            "long",
            "short",
            "uint64",
            "uint32",
            "uint16",
            "uint8",
            "complex128",
            "complex64",
            "bool_",
            "newaxis",
        }

        if name in bypass_list:
            return getattr(real_np, name)

        return getattr(self._numpy, name)


if TYPE_CHECKING:
    import numpy as anp
else:
    anp = DifferentiationProxy()
