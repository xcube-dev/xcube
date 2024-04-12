# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Mapping, Sequence, Hashable, Dict

import numpy as np
import pandas as pd
import xarray as xr
from deprecated import deprecated

from xcube.core.schema import get_dataset_xy_var_names
from xcube.util.timeindex import ensure_time_label_compatible

# Exported for backward compatibility only
# noinspection PyUnresolvedReferences
from ._tile2 import (
    compute_tiles,
    compute_rgba_tile,
    get_var_valid_range,
    get_var_cmap_params,
    DEFAULT_VALUE_RANGE,
    DEFAULT_FORMAT,
    DEFAULT_CRS_NAME,
    DEFAULT_TILE_SIZE,
    DEFAULT_CMAP_NAME,
    DEFAULT_TILE_ENLARGEMENT,
    TileNotFoundException,
    TileRequestException,
)


@deprecated(reason="Function is obsolete.", version="0.13.0")
def parse_non_spatial_labels(
    raw_labels: Mapping[str, str],
    dims: Sequence[Hashable],
    coords: Mapping[Hashable, xr.DataArray],
    allow_slices: bool = False,
    exception_type: type = ValueError,
    var: xr.DataArray = None,
) -> Mapping[str, Any]:
    xy_var_names = get_dataset_xy_var_names(coords, must_exist=False)
    if xy_var_names is None:
        raise exception_type(f"missing spatial coordinates")
    xy_dims = set(coords[xy_var_name].dims[0] for xy_var_name in xy_var_names)

    # noinspection PyShadowingNames
    def to_datetime(datetime_str: str, dim_var: xr.DataArray):
        if datetime_str == "current":
            return dim_var[-1]
        else:
            return pd.to_datetime(datetime_str)

    parsed_labels = {}
    for dim in dims:
        if dim in xy_dims:
            continue
        dim_var = coords[dim]
        label_str = raw_labels.get(str(dim))
        try:
            if label_str is None:
                label = dim_var[0].values
            elif label_str == "current":
                label = dim_var[-1].values
            else:
                if "/" in label_str:
                    label_strs = tuple(label_str.split("/", maxsplit=1))
                else:
                    label_strs = (label_str,)
                if np.issubdtype(dim_var.dtype, np.floating):
                    labels = tuple(map(float, label_strs))
                elif np.issubdtype(dim_var.dtype, np.integer):
                    labels = tuple(map(int, label_strs))
                elif np.issubdtype(dim_var.dtype, np.datetime64):
                    labels = tuple(to_datetime(label, dim_var) for label in label_strs)
                else:
                    raise exception_type(
                        f"unable to parse value"
                        f" {label_str!r}"
                        f" into a {dim_var.dtype!r}"
                    )
                if len(labels) == 1:
                    label = labels[0]
                else:
                    if allow_slices:
                        # noinspection PyTypeChecker
                        label = slice(labels[0], labels[1])
                    elif np.issubdtype(dim_var.dtype, np.integer):
                        label = labels[0] + (labels[1] - labels[0]) // 2
                    else:
                        label = labels[0] + (labels[1] - labels[0]) / 2
            parsed_labels[str(dim)] = label
        except ValueError as e:
            raise exception_type(
                f"{label_str!r} is not a valid" f" value for dimension {dim!r}"
            ) from e

    if var is not None:
        return ensure_time_label_compatible(var, parsed_labels)
    else:
        return parsed_labels
