# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Callable

import xarray as xr

from xcube.util.assertions import assert_instance


class ExprVar:
    """A wrapped `xarray.DataArray` to allow safe access in expressions.

    Args:
        da: The `xarray.DataArray` to be wrapped.
    """

    def __init__(self, da: xr.DataArray):
        assert_instance(da, xr.DataArray, name="da")
        # Note that the double underscore protects access by "name mangling"
        self.__da = da

    ################################################
    # Binary operations - comparisons

    def __eq__(self, other):
        return self.__wrap(self.__da == self.__unwrap(other))

    def __ne__(self, other):
        return self.__wrap(self.__da != self.__unwrap(other))

    def __le__(self, other):
        return self.__wrap(self.__da <= self.__unwrap(other))

    def __lt__(self, other):
        return self.__wrap(self.__da < self.__unwrap(other))

    def __ge__(self, other):
        return self.__wrap(self.__da >= self.__unwrap(other))

    def __gt__(self, other):
        return self.__wrap(self.__da > self.__unwrap(other))

    ################################################
    # Binary operations - emulating numeric type

    def __add__(self, other):
        return self.__wrap(self.__da + self.__unwrap(other))

    def __radd__(self, other):
        return self.__wrap(self.__unwrap(other) + self.__da)

    def __sub__(self, other):
        return self.__wrap(self.__da - self.__unwrap(other))

    def __rsub__(self, other):
        return self.__wrap(self.__unwrap(other) - self.__da)

    def __mul__(self, other):
        return self.__wrap(self.__da * self.__unwrap(other))

    def __rmul__(self, other):
        return self.__wrap(self.__unwrap(other) * self.__da)

    def __truediv__(self, other):
        return self.__wrap(self.__da / self.__unwrap(other))

    def __rtruediv__(self, other):
        return self.__wrap(self.__unwrap(other) / self.__da)

    def __floordiv__(self, other):
        return self.__wrap(self.__da // self.__unwrap(other))

    def __rfloordiv__(self, other):
        return self.__wrap(self.__unwrap(other) // self.__da)

    def __mod__(self, other):
        return self.__wrap(self.__da % self.__unwrap(other))

    def __rmod__(self, other):
        return self.__wrap(self.__unwrap(other) % self.__da)

    def __pow__(self, power):
        return self.__wrap(self.__da ** self.__unwrap(power))

    def __rpow__(self, other):
        return self.__wrap(self.__unwrap(other) ** self.__da)

    def __lshift__(self, other):
        return self.__wrap(self.__da << self.__unwrap(other))

    def __rlshift__(self, other):
        # Not supported by xarray, will raise
        return self.__wrap(self.__unwrap(other) << self.__da)

    def __rshift__(self, other):
        return self.__wrap(self.__da >> self.__unwrap(other))

    def __rrshift__(self, other):
        # Not supported by xarray, will raise
        return self.__wrap(self.__unwrap(other) >> self.__da)

    def __and__(self, other):
        return self.__wrap(self.__da & self.__unwrap(other))

    def __rand__(self, other):
        return self.__wrap(self.__unwrap(other) & self.__da)

    def __xor__(self, other):
        return self.__wrap(self.__da ^ self.__unwrap(other))

    def __rxor__(self, other):
        return self.__wrap(self.__unwrap(other) ^ self.__da)

    def __or__(self, other):
        return self.__wrap(self.__da | self.__unwrap(other))

    def __ror__(self, other):
        return self.__wrap(self.__unwrap(other) | self.__da)

    ################################################
    # Unary operations

    def __pos__(self):
        return self.__wrap(+self.__da)

    def __neg__(self):
        return self.__wrap(-self.__da)

    def __invert__(self):
        return self.__wrap(~self.__da)

    def __abs__(self):
        return self.__wrap(abs(self.__da))

    ################################################
    # Internal helpers

    @staticmethod
    def __unwrap(v):
        return v.__da if isinstance(v, ExprVar) else v

    @staticmethod
    def __wrap(v):
        return ExprVar(v) if isinstance(v, xr.DataArray) else v

    @staticmethod
    def _wrap_fn(fn: Callable) -> Callable:
        def wrapped_fn(*args, **kwargs):
            return ExprVar.__wrap(
                fn(
                    *(ExprVar.__unwrap(arg) for arg in args),
                    **{kw: ExprVar.__unwrap(arg) for kw, arg in kwargs.items()},
                )
            )

        return wrapped_fn
