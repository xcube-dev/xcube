# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import inspect
from typing import Any, Callable, Union

import numpy as np
import xarray as xr

from .error import VarExprError
from .exprvar import ExprVar
from .names import (
    _GLOBAL_NAMES,
    get_xarray_funcs,
    get_numpy_ufuncs,
    get_constants,
)
from .varexpr import evaluate


class VarExprContext:
    """Allow safe evaluation of expressions in the context of a
    `xarray.Dataset` object.

    Args:
        dataset: The dataset that provides the variables that can
            be accessed in the expressions passed to :meth:`evaluate`.
    """

    def __init__(self, dataset: xr.Dataset):
        self._array_variables = dict(
            {str(k): ExprVar(v) for k, v in dataset.data_vars.items()}
        )
        self._names = dict(_GLOBAL_NAMES)
        self._names.update(self._array_variables)

    def names(self) -> dict[str, Any]:
        return dict(self._names)

    def get_array_variables(self) -> list[str]:
        """Get array functions."""
        return sorted(self._array_variables.keys())

    @classmethod
    def get_array_functions(cls) -> list[str]:
        """Get array functions."""
        # noinspection PyTypeChecker
        return sorted(
            _format_callable(name, fn)
            for name, fn in dict(**get_xarray_funcs(), **get_numpy_ufuncs()).items()
        )

    @classmethod
    def get_other_functions(cls) -> list[str]:
        """Get other functions that cannot be used with arrays."""
        return []

    @classmethod
    def get_array_operators(cls) -> list[str]:
        """Get array operators."""
        return [
            "+X",
            "-X",
            "~X",
            "X + Y",
            "X - Y",
            "X * Y",
            "X / Y",
            "X // Y",
            "X % Y",
            "X ** Y",
            "X << Y",
            "X >> Y",
            "X & Y",
            "X ^ Y",
            "X | Y",
            "X == Y",
            "X != Y",
            "X < Y",
            "X <= Y",
            "X > Y",
            "X >= Y",
        ]

    @classmethod
    def get_other_operators(cls) -> list[str]:
        """Get other operators that cannot be used with arrays."""
        return [
            "X and Y",
            "X or Y",
            "not X",
            "X in Y",
            "X not in Y",
            "X is Y",
            "X is not Y",
        ]

    @classmethod
    def get_constants(cls) -> list[str]:
        """Get constants."""
        return sorted(get_constants().keys())

    def evaluate(self, var_expr: str) -> xr.DataArray:
        """Evaluate given Python expression *var_expr* in the context of an
        `xarray.Dataset` object.

        The expression *var_expr* may reference the following names:

        * the dataset's data variables;
        * the dataset's coordinate variables;
        * the numpy constants `e`, `pi`, `nan`, `inf`;
        * all numpy ufuncs (https://numpy.org/doc/stable/reference/ufuncs.html);
        * the `where` function (https://docs.xarray.dev/en/stable/generated/xarray.where.html);
        * the Python built-in functions `min`, `max`, `round`, `floor`, `ceil`,
          `bool`, `int`, `float`, `complex`, `str`, `tuple`, `set`, `list`, `dict`.

        In general, all Python numerical and logical operators such as
        `not`, `and`, `or` are supported.
        However, for dataset variables the following subset of operators apply:

        * binary comparison operators: `==`, `!=`, `<`, `<=`, `>`, `>=`;
        * binary arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`,
            `**`, `<<`, `>>`, `&`, `^`, `|`;
        * unary operators: `+`, `-`, `~`.

        Args:
            var_expr: Expression to be evaluated.

        Returns:
            A newly computed variable of type `xarray.DataArray`.
        """
        try:
            result = evaluate(var_expr, self._names)
        except BaseException as e:
            # Do not report the name 'ExprVar'
            raise VarExprError(f"{e}".replace("ExprVar", "DataArray")) from e

        if not isinstance(result, ExprVar):
            # We do not mention 'ExprVar' by intention
            raise VarExprError(
                f"result must be a 'DataArray' object,"
                f" but got type {result.__class__.__name__!r}"
            )

        result = result.__dict__.get("_ExprVar__da")
        if not isinstance(result, xr.DataArray):
            # noinspection PyUnresolvedReferences
            raise RuntimeError(
                f"internal error: 'DataArray' object expected,"
                f" but got type {result.__class__.__name__!r}"
            )

        return result


def _format_callable(name: str, fn: Union[np.ufunc, Callable]):
    if name == "where":
        return "where(C,X,Y)"
    if isinstance(fn, np.ufunc):
        num_args = fn.nargs - 1
    else:
        signature = inspect.signature(fn)
        num_args = len(signature.parameters.values())
    if num_args == 0:
        return f"{name}()"
    elif num_args == 1:
        return f"{name}(X)"
    elif num_args == 2:
        return f"{name}(X,Y)"
    elif num_args == 3:
        return f"{name}(X,Y,Z)"
    else:
        return f"{name}({','.join(map(lambda i: f'X{i + 1}', range(num_args)))})"
