# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Callable

import numpy as np
import xarray as xr

from .exprvar import ExprVar


def get_constants() -> dict[str, Any]:
    return {
        "nan": np.nan,
        "e": np.e,
        "inf": np.inf,
        "pi": np.pi,
    }


def get_numpy_ufuncs() -> dict[str, np.ufunc]:
    """Get numpy universal functions (considered safe)"""
    # noinspection PyProtectedMember
    return {
        k: v
        for k, v in np.__dict__.items()
        if isinstance(v, np.ufunc) and isinstance(k, str) and not k.startswith("_")
    }


def get_xarray_funcs() -> dict[str, Callable]:
    """Get safe xarray functions"""
    # noinspection PyProtectedMember
    return {"where": xr.where}


# noinspection PyProtectedMember
_GLOBAL_NAMES = {
    **get_constants(),
    **{k: ExprVar._wrap_fn(fn) for k, fn in get_numpy_ufuncs().items()},
    **{k: ExprVar._wrap_fn(fn) for k, fn in get_xarray_funcs().items()},
}

# noinspection PyProtectedMember
del ExprVar._wrap_fn
