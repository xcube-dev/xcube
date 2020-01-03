import os
from typing import Set

import xarray as xr


def open_sentinel3_product(path: str, var_names: Set[str] = None) -> xr.Dataset:
    """
    Open a Sentinel-3 product from given path.

    :param path: Sentinel-3 product path
    :param var_names: Optional variable names to be included.
    :return: A dataset representation of the Sentinel-3 product.
    """
    x_name = 'longitude'
    y_name = 'latitude'
    data_vars = {}
    geo_vars_file_name = 'geo_coordinates.nc'
    file_names = set(file_name for file_name in os.listdir(path) if file_name.endswith('.nc'))
    if geo_vars_file_name not in file_names:
        raise ValueError(f'missing file {geo_vars_file_name!r} in {path}')
    file_names.remove(geo_vars_file_name)
    geo_vars_path = os.path.join(path, geo_vars_file_name)
    with xr.open_dataset(geo_vars_path) as geo_ds:
        if x_name not in geo_ds:
            raise ValueError(f'variable {x_name!r} not found in {geo_vars_path}')
        if y_name not in geo_ds:
            raise ValueError(f'variable {y_name!r} not found in {geo_vars_path}')
        x_var = geo_ds[x_name]
        y_var = geo_ds[y_name]
        if x_var.ndim != 2:
            raise ValueError(f'variable {x_name!r} must have two dimensions')
        if y_var.ndim != x_var.ndim \
                or y_var.shape != x_var.shape \
                or y_var.dims != x_var.dims:
            raise ValueError(f'variable {y_name!r} must have same shape and dimensions as {x_name!r}')
        data_vars.update({x_name: x_var, y_name: y_var})
    for file_name in file_names:
        with xr.open_dataset(os.path.join(path, file_name)) as ds:
            for var_name, var in ds.data_vars.items():
                if var_names and var_name not in var_names:
                    continue
                if var.ndim >= 2 \
                        and var.shape[-2:] == x_var.shape \
                        and var.dims[-2:] == x_var.dims:
                    data_vars.update({var_name: var})
    return xr.Dataset(data_vars)


def is_sentinel3_product(path: str) -> bool:
    """
    Test if given *path* is likely a Sentinel-3 product path.

    :param path: (directory) path
    :return: True, if so
    """
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, 'geo_coordinates.nc'))
