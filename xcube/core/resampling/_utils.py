# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Callable, Literal, TypeAlias

import numpy as np

AggMethod: TypeAlias = Literal[
    "center",
    "count",
    "first",
    "last",
    "max",
    "mean",
    "median",
    "mode",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]
AggMethods: TypeAlias = AggMethod | dict[AggMethod, list[str | np.dtype]]

AggFunction: TypeAlias = Callable[[np.ndarray, tuple[int, ...] | None], np.ndarray]

AGG_METHODS: dict[AggMethod, AggFunction] = {
    "center": xec.center,
    "count": np.count_nonzero,
    "first": xec.first,
    "last": xec.last,
    "prod": np.nanprod,
    "max": np.nanmax,
    "mean": xec.mean,
    "median": xec.median,
    "min": np.nanmin,
    "mode": xec.mode,
    "std": xec.std,
    "sum": np.nansum,
    "var": xec.var,
}

SplineOrder: TypeAlias = Literal[0, 1, 2, 3]
SplineOrders: TypeAlias = SplineOrder | dict[SplineOrder, list[str | np.dtype]]
