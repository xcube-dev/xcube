# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import fnmatch
from collections.abc import Hashable, Mapping
from typing import List, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr
from scipy import stats

from xcube.util.assertions import assert_in, assert_instance, assert_true

AGG_METHODS = "auto", "first", "min", "max", "mean", "median", "mode"
DEFAULT_INT_AGG_METHOD = "first"
DEFAULT_FLOAT_AGG_METHOD = "mean"

AggMethod = Union[None, str]
AggMethods = Union[AggMethod, Mapping[str, AggMethod]]


def subsample_dataset(
    dataset: xr.Dataset,
    step: int,
    xy_dim_names: Optional[tuple[str, str]] = None,
    agg_methods: Optional[AggMethods] = None,
) -> xr.Dataset:
    """Subsample *dataset* with given integer subsampling *step*.
    Only data variables with spatial dimensions given by
    *xy_dim_names* are subsampled.

    Args:
        dataset: the dataset providing the variables
        step: the integer subsampling step size in pixels in
            the x and y directions.
            For aggregation methods other than "first" it defines the
            window size for the aggregation.
        xy_dim_names: the spatial dimension names
        agg_methods: Optional aggregation methods.
            May be given as string or as mapping from variable name pattern
            to aggregation method. Valid aggregation methods are
            "auto", "first", "min", "max", "mean", "median", and "mode".
            If "auto", the default, "first" is used for integer variables
            and "mean" for floating point variables.
    Returns:
        The subsampled dataset or a tuple comprising the
        subsampled dataset and the effective aggregation methods.
    """
    assert_instance(dataset, xr.Dataset, name="dataset")
    assert_instance(step, int, name="step")
    assert_valid_agg_methods(agg_methods)

    x_name, y_name = xy_dim_names or ("y", "x")

    agg_methods = get_dataset_agg_methods(
        dataset, xy_dim_names=xy_dim_names, agg_methods=agg_methods
    )

    new_data_vars = dict()
    new_coords = None  # used to collect coordinates from coarsen
    for var_name, var in dataset.data_vars.items():
        if var_name in agg_methods:
            agg_method = agg_methods[var_name]
            if agg_method == "first":
                slices = get_variable_subsampling_slices(
                    var, step, xy_dim_names=xy_dim_names
                )
                assert slices is not None
                new_var = var[slices]
            else:
                if agg_method == "mode":
                    new_var: xr.DataArray = _agg_mode(var, x_name, y_name, step)
                else:
                    new_var: xr.DataArray = _agg_builtin(
                        var, x_name, y_name, step, agg_method
                    )
                if new_var.dtype != var.dtype:
                    # We don't want, e.g. "mean", to turn data
                    # from dtype unit16 into float64
                    new_var = new_var.astype(var.dtype)
                new_var.attrs.update(var.attrs)
                new_var.encoding.update(var.encoding)
                # coarsen() recomputes spatial coordinates.
                # Collect them, so we can later apply them to the
                # variables that are subsampled by "first"
                # (= slice selection).
                if new_coords is None:
                    new_coords = dict(new_var.coords)
                else:
                    new_coords.update(new_var.coords)
        else:
            new_var = var

        new_data_vars[var_name] = new_var

    if not new_data_vars:
        return dataset

    if new_coords:
        # Make sure all variables use the same modified
        # spatial coordinates from coarsen
        new_data_vars = {
            k: v.assign_coords({d: new_coords[d] for d in v.dims if d in new_coords})
            for k, v in new_data_vars.items()
        }

    return xr.Dataset(data_vars=new_data_vars, attrs=dataset.attrs)


def _agg_mode(var: xr.DataArray, x_name: str, y_name: str, step: int) -> xr.DataArray:
    var_coarsen = _coarsen(var, x_name, y_name, step)
    drop_axis = _get_drop_axis(var, x_name, y_name, step)

    def _scipy_mode(x, axis, **kwargs):
        return stats.mode(x, axis, nan_policy="omit", **kwargs).mode

    def _mode_dask(x, axis, **kwargs):
        return x.map_blocks(
            _scipy_mode, axis=axis, dtype=x.dtype, drop_axis=drop_axis, **kwargs
        )

    if isinstance(var.data, da.Array):
        return var_coarsen.reduce(_mode_dask)
    else:
        return var_coarsen.reduce(_scipy_mode)


def _agg_builtin(
    var: xr.DataArray, x_name: str, y_name: str, step: int, agg_method: str
) -> xr.DataArray:
    var_coarsen = _coarsen(var, x_name, y_name, step)
    return getattr(var_coarsen, agg_method)()


def _coarsen(var: xr.DataArray, x_name: str, y_name: str, step: int):
    dim = dict()
    if x_name in var.dims:
        dim[x_name] = step
    if y_name in var.dims:
        dim[y_name] = step
    return var.coarsen(dim=dim, boundary="pad", coord_func="min")


def _get_drop_axis(var: xr.DataArray, x_name: str, y_name: str, step: int) -> List[int]:
    """
    This function serves to determine the indexes of the dimensions of a coarsened
    array that will not be included in the array after reduction. These dimensions
    are the indexes following the indexes of the spatial dimensions from the original
    array, so first these indexes are determined, then they are shifted appropriately.
    """
    drop_axis = []
    if x_name in var.dims:
        drop_axis.append(var.dims.index(x_name))
    if y_name in var.dims:
        drop_axis.append(var.dims.index(y_name))
    if len(drop_axis) == 1:
        drop_axis[0] += 1
    elif len(drop_axis) == 2:
        # The latter index must be shifted by two positions
        if drop_axis[0] > drop_axis[1]:
            drop_axis[0] += 2
            drop_axis[1] += 1
        else:
            drop_axis[0] += 1
            drop_axis[1] += 2
    return drop_axis


def get_dataset_agg_methods(
    dataset: xr.Dataset,
    xy_dim_names: Optional[tuple[str, str]] = None,
    agg_methods: Optional[AggMethods] = None,
):
    assert_instance(dataset, xr.Dataset, name="dataset")
    assert_valid_agg_methods(agg_methods)

    x_name, y_name = xy_dim_names or ("y", "x")

    dataset_agg_methods = dict()

    for var_name, var in dataset.data_vars.items():
        if x_name in var.dims or y_name in var.dims:
            agg_method = find_agg_method(agg_methods, var_name, var.dtype)
            dataset_agg_methods[str(var_name)] = agg_method

    return dataset_agg_methods


def assert_valid_agg_methods(agg_methods: Optional[AggMethods]):
    """Assert that the given *agg_methods* are valid."""
    assert_instance(
        agg_methods, (type(None), str, collections.abc.Mapping), name="agg_methods"
    )
    if isinstance(agg_methods, str):
        assert_in(agg_methods, AGG_METHODS, name="agg_methods")
    elif agg_methods is not None:
        enum = (None, *AGG_METHODS)
        for k, v in agg_methods.items():
            assert_true(
                isinstance(k, str), message="keys in agg_methods must be strings"
            )
            assert_true(
                v in enum, message=f"values in agg_methods must be one of {enum}"
            )


def find_agg_method(
    agg_methods: AggMethods, var_name: Hashable, var_dtype: np.dtype
) -> str:
    """Find aggregation method in *agg_methods*
    for given *var_name* and *var_dtype*.
    """
    assert_valid_agg_methods(agg_methods)
    if isinstance(agg_methods, str) and agg_methods != "auto":
        return agg_methods
    if isinstance(agg_methods, collections.abc.Mapping):
        for var_name_pat, agg_method in agg_methods.items():
            if var_name == var_name_pat or fnmatch.fnmatch(str(var_name), var_name_pat):
                if agg_method in (None, "auto"):
                    break
                return agg_method
    # here: agg_method is either None or 'auto'
    if np.issubdtype(var_dtype, np.integer):
        return "first"
    else:
        return "mean"


_FULL_SLICE = slice(None, None, None)


def get_variable_subsampling_slices(
    variable: xr.DataArray, step: int, xy_dim_names: Optional[tuple[str, str]] = None
) -> Optional[tuple[slice, ...]]:
    """Compute subsampling array slices for *variable*.

    Args:
        variable: the dataset providing the variables
        step: the integer subsampling step
        xy_dim_names: the spatial dimension names

    Returns:
        A tuple that comprises array slices for all variable dimensions.
        The slices for spatial dimensions are subsampled according
        to *step*, while other slices are cover every array item.
        Returns `None`, if *variable* does not contain spatial
        dimensions at all.
    """
    assert_instance(variable, xr.DataArray, name="variable")
    assert_instance(step, int, name="step")
    x_dim_name, y_dim_name = xy_dim_names or ("x", "y")

    var_index = None
    for index, dim_name in enumerate(variable.dims):
        if dim_name == x_dim_name or dim_name == y_dim_name:
            if var_index is None:
                var_index = index * [_FULL_SLICE]
            var_index.append(slice(None, None, step))
        elif var_index is not None:
            var_index.append(_FULL_SLICE)
    return tuple(var_index) if var_index is not None else None
