# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os
from typing import Set, Tuple, Union
from collections.abc import Mapping

import xarray as xr


def open_sentinel3_product(
    path: str,
    var_names: set[str] = None,
    chunks: Mapping[str, Union[int, tuple[int, ...]]] = None,
) -> xr.Dataset:
    """Open a Sentinel-3 product from given path.

    Args:
        chunks: Optional mapping from dimension name to chunk sizes or
            chunk size tuples.
        path: Sentinel-3 product path
        var_names: Optional variable names to be included.

    Returns:
        A dataset representation of the Sentinel-3 product.
    """
    x_name = "longitude"
    y_name = "latitude"
    data_vars = {}
    geo_vars_file_name = "geo_coordinates.nc"
    file_names = {
        file_name for file_name in os.listdir(path) if file_name.endswith(".nc")
    }
    if geo_vars_file_name not in file_names:
        raise ValueError(f"missing file {geo_vars_file_name!r} in {path}")
    file_names.remove(geo_vars_file_name)
    geo_vars_path = os.path.join(path, geo_vars_file_name)
    with _open_dataset(geo_vars_path, chunks=chunks) as geo_ds:
        if x_name not in geo_ds:
            raise ValueError(f"variable {x_name!r} not found in {geo_vars_path}")
        if y_name not in geo_ds:
            raise ValueError(f"variable {y_name!r} not found in {geo_vars_path}")
        x_var = geo_ds[x_name]
        y_var = geo_ds[y_name]
        if x_var.ndim != 2:
            raise ValueError(f"variable {x_name!r} must have two dimensions")
        if (
            y_var.ndim != x_var.ndim
            or y_var.shape != x_var.shape
            or y_var.dims != x_var.dims
        ):
            raise ValueError(
                f"variable {y_name!r} must have same shape and dimensions as {x_name!r}"
            )
        data_vars.update({x_name: x_var, y_name: y_var})
    for file_name in file_names:
        dataset_path = os.path.join(path, file_name)
        with _open_dataset(dataset_path, chunks=chunks) as ds:
            for var_name, var in ds.data_vars.items():
                if var_names and var_name not in var_names:
                    continue
                if (
                    var.ndim >= 2
                    and var.shape[-2:] == x_var.shape
                    and var.dims[-2:] == x_var.dims
                ):
                    data_vars.update({var_name: var})
    return xr.Dataset(data_vars)


def _open_dataset(dataset_path, chunks=None) -> xr.Dataset:
    if chunks is None:
        ds = xr.open_dataset(dataset_path)
        for var_name, var in ds.variables.items():
            chunk_sizes = var.encoding.get("chunksizes")
            if isinstance(chunk_sizes, (tuple, list)) and len(chunk_sizes) == len(
                var.dims
            ):
                if chunks is None:
                    chunks = dict()
                chunks.update({dim: size for dim, size in zip(var.dims, chunk_sizes)})
        if chunks is None:
            return ds
    return xr.open_dataset(dataset_path, chunks=chunks)


def is_sentinel3_product(path: str) -> bool:
    """Test if given *path* is likely a Sentinel-3 product path.

    Args:
        path: (directory) path

    Returns:
        True, if so
    """
    return os.path.isdir(path) and os.path.isfile(
        os.path.join(path, "geo_coordinates.nc")
    )
