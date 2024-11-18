# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import numpy as np
import xarray as xr


def is_empty_cube(cube: xr.Dataset) -> bool:
    return len(cube.data_vars) == 0


def strip_cube(cube: xr.Dataset) -> xr.Dataset:
    drop_vars = [
        k
        for k, v in cube.data_vars.items()
        if len(v.shape) < 3
        or np.prod(v.shape) == 0
        or v.shape[-2] < 2
        or v.shape[-1] < 2
    ]
    if drop_vars:
        return cube.drop_vars(drop_vars)
    return cube
