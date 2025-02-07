# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import numpy as np
import pandas as pd
import xarray as xr


def get_test_dataset(var_names=("noise",)) -> xr.Dataset:
    w = 1440
    h = 720
    p = 14

    x1 = -180.0
    y1 = -90.0
    x2 = +180.0
    y2 = +90.0
    dx = (x2 - x1) / w
    dy = (y2 - y1) / h

    data_vars = {
        var_name: (("time", "lat", "lon"), np.random.rand(p, h, w))
        for var_name in var_names
    }

    lat_bnds = np.array(
        list(zip(np.linspace(y2, y1 + dy, num=h), np.linspace(y2 - dy, y1, num=h)))
    )
    lon_bnds = np.array(
        list(zip(np.linspace(x1, x2 - dx, num=w), np.linspace(x1 + x2, x2, num=w)))
    )
    coords = dict(
        time=(
            ("time",),
            np.array(
                pd.date_range(start="2019-01-01T12:00:00Z", periods=p, freq="1D"),
                dtype="datetime64[ns]",
            ),
        ),
        lat=(("lat",), np.linspace(y2 - 0.5 * dy, y1 + 0.5 * dy, num=h)),
        lon=(("lon",), np.linspace(x1 + 0.5 * dx, x2 - 0.5 * dx, num=w)),
        lat_bnds=(("lat", "bnds"), lat_bnds),
        lon_bnds=(("lon", "bnds"), lon_bnds),
    )

    return xr.Dataset(coords=coords, data_vars=data_vars).chunk(
        chunks=dict(lat=180, lon=180)
    )
