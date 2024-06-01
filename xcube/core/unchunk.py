# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json
import os.path
from typing import List
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

    is_zarr = os.path.isfile(os.path.join(dataset_path, ".zgroup"))
    if not is_zarr:
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
    for var_name in var_names:
        var_path = os.path.join(dataset_path, var_name)

        # Optimization: if "shape" and "chunks" are equal in ${var}/.zarray, we are done
        var_array_info_path = os.path.join(var_path, ".zarray")
        with open(var_array_info_path) as fp:
            var_array_info = json.load(fp)
            if var_array_info.get("shape") == var_array_info.get("chunks"):
                continue

        # Open array and remove chunks from the data
        var_array = zarr.convenience.open_array(var_path, "r+")
        if var_array.shape != var_array.chunks:
            # TODO (forman): Fully loading data is inefficient and dangerous for large arrays.
            #                Instead save unchunked to temp and replace existing chunked array dir with temp.
            # Fully load data and attrs so we no longer depend on files
            data = np.array(var_array)
            attributes = var_array.attrs.asdict()
            # Save array data
            zarr.convenience.save_array(
                var_path, data, chunks=False, fill_value=var_array.fill_value
            )
            # zarr.convenience.save_array() does not seem save user attributes (file ".zattrs" not written),
            # therefore we must modify attrs explicitly:
            var_array = zarr.convenience.open_array(var_path, "r+")
            var_array.attrs.update(attributes)

    zarr.consolidate_metadata(dataset_path)
