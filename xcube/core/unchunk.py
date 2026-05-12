# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json
import os.path
from collections.abc import Sequence

import numpy as np
import xarray as xr
import zarr


def unchunk_dataset(
    dataset_path: str, var_names: Sequence[str] = None, coords_only: bool = False
):
    """Unchunk dataset variables in-place.

    Args:
        dataset_path: Path to ZARR dataset directory.
        var_names: Optional list of variable names.
        coords_only: Un-chunk coordinate variables only.
    """

    if not (
        os.path.isfile(os.path.join(dataset_path, ".zgroup"))
        or os.path.isfile(os.path.join(dataset_path, "zarr.json"))
    ):
        raise ValueError(f"{dataset_path!r} is not a valid Zarr directory")

    with xr.open_zarr(dataset_path) as dataset:
        if var_names is None:
            if coords_only:
                var_names = list(dataset.coords)
            else:
                var_names = list(dataset.variables)
        else:
            for var_name in var_names:
                if coords_only:
                    if var_name not in dataset.coords:
                        raise ValueError(
                            f"variable {var_name!r} is not a coordinate variable in {dataset_path!r}"
                        )
                else:
                    if var_name not in dataset.variables:
                        raise ValueError(
                            f"variable {var_name!r} is not a variable in {dataset_path!r}"
                        )

    _unchunk_vars(dataset_path, var_names)


def _unchunk_vars(dataset_path: str, var_names: list[str]):
    root = zarr.open_group(dataset_path, mode="a")

    for var_name in var_names:
        arr = root[var_name]

        # already unchunked
        if tuple(arr.shape) == tuple(arr.chunks):
            continue

        # fully load data
        data = np.asarray(arr)

        # preserve metadata
        attrs = dict(arr.attrs)
        fill_value = arr.fill_value

        # Zarr v3 metadata
        dimension_names = getattr(arr.metadata, "dimension_names", None)
        compressors = getattr(arr.metadata, "compressors", None)
        filters = getattr(arr.metadata, "filters", None)

        # remove old array completely
        del root[var_name]

        # recreate as single chunk
        new_arr = root.create_array(
            name=var_name,
            data=data,
            chunks=data.shape,
            fill_value=fill_value,
            dimension_names=dimension_names,
            compressors=compressors,
            filters=filters,
        )

        # restore attrs
        new_arr.attrs.update(attrs)

    zarr.consolidate_metadata(dataset_path)
