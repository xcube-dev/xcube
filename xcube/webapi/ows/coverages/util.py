# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


import xarray as xr


def get_h_dim(ds: xr.Dataset) -> str:
    return [d for d in list(map(str, ds.sizes)) if d[:3].lower() in {"x", "lon"}][0]


def get_v_dim(ds: xr.Dataset) -> str:
    return [d for d in list(map(str, ds.sizes)) if d[:3].lower() in {"y", "lat"}][0]
