# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Dict, Union, Tuple, Any, Optional
from collections.abc import Mapping, Sequence, Hashable

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.verify import assert_cube

DEFAULT_INDEX_NAME_PATTERN = "{name}_index"
DEFAULT_REF_NAME_PATTERN = "{name}_ref"
INDEX_DIM_NAME = "idx"

POINT_INTERP_METHOD_NEAREST = "nearest"
POINT_INTERP_METHOD_LINEAR = "linear"
DEFAULT_INTERP_POINT_METHOD = POINT_INTERP_METHOD_NEAREST

SeriesLike = Union[
    np.ndarray, da.Array, xr.DataArray, pd.Series, Sequence[Union[int, float]]
]

PointsLike = Union[xr.Dataset, pd.DataFrame, Mapping[str, SeriesLike]]


def get_cube_values_for_points(
    cube: xr.Dataset,
    points: PointsLike,
    var_names: Sequence[str] = None,
    include_coords: bool = False,
    include_bounds: bool = False,
    include_indexes: bool = False,
    index_name_pattern: str = DEFAULT_INDEX_NAME_PATTERN,
    include_refs: bool = False,
    ref_name_pattern: str = DEFAULT_REF_NAME_PATTERN,
    method: str = DEFAULT_INTERP_POINT_METHOD,
    cube_asserted: bool = False,
) -> xr.Dataset:
    """Extract values from *cube* variables at given
    coordinates in *points*.

    Returns a new dataset with values of variables from *cube* selected
    by the coordinate columns
    provided in *points*. All variables will be 1-D and have the same order
    as the rows in points.

    Args:
        cube: The cube dataset.
        points: Dictionary that maps dimension name to coordinate
            arrays.
        var_names: An optional list of names of data variables in *cube*
            whose values shall be extracted.
        include_coords: Whether to include the cube coordinates for each
            point in return value.
        include_bounds: Whether to include the cube coordinate
            boundaries (if any) for each point in return value.
        include_indexes: Whether to include computed indexes into the
            cube for each point in return value.
        index_name_pattern: A naming pattern for the computed index
            columns. Must include "{name}" which will be replaced by the
            index' dimension name.
        include_refs: Whether to include point (reference) values from
            *points* in return value.
        ref_name_pattern: A naming pattern for the computed point data
            columns. Must include "{name}" which will be replaced by the
            point's attribute name.
        method: "nearest" or "linear".
        cube_asserted: If False, *cube* will be verified, otherwise it
            is expected to be a valid cube.

    Returns:
        A new data frame whose columns are values from *cube* variables
        at given *points*.
    """
    if not cube_asserted:
        assert_cube(cube)

    index_dtype = np.int64 if method == POINT_INTERP_METHOD_NEAREST else np.float64

    point_indexes = get_cube_point_indexes(
        cube,
        points,
        index_name_pattern=index_name_pattern,
        index_dtype=index_dtype,
        cube_asserted=True,
    )

    cube_values = get_cube_values_for_indexes(
        cube,
        point_indexes,
        include_coords,
        include_bounds,
        data_var_names=var_names,
        index_name_pattern=index_name_pattern,
        method=method,
        cube_asserted=True,
    )

    if include_indexes:
        cube_values.update(point_indexes)

    if include_refs:
        point_refs = xr.Dataset(
            {
                ref_name_pattern.format(name=name): xr.DataArray(
                    points[name], dims=[INDEX_DIM_NAME]
                )
                for name in points.keys()
            }
        )

        cube_values.update(point_refs)

    # The (coordinate) variable named INDEX_DIM_NAME should not exist,
    # otherwise cube_values.to_dataframe() will use an empty string
    # to name that series "" in the output DataFrame instance.
    # This seems to be an issue of xarray 0.18.2+.
    if INDEX_DIM_NAME in cube_values:
        cube_values = cube_values.drop_vars(INDEX_DIM_NAME)

    return cube_values


def get_cube_values_for_indexes(
    cube: xr.Dataset,
    indexes: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
    include_coords: bool = False,
    include_bounds: bool = False,
    data_var_names: Sequence[str] = None,
    index_name_pattern: str = DEFAULT_INDEX_NAME_PATTERN,
    method: str = DEFAULT_INTERP_POINT_METHOD,
    cube_asserted: bool = False,
) -> xr.Dataset:
    """Get values from the *cube* at given *indexes*.

    Args:
        cube: A cube dataset.
        indexes: A mapping from column names to index and fraction
            arrays for all cube dimensions.
        include_coords: Whether to include the cube coordinates for each
            point in return value.
        include_bounds: Whether to include the cube coordinate
            boundaries (if any) for each point in return value.
        data_var_names: An optional list of names of data variables in
            *cube* whose values shall be extracted.
        index_name_pattern: A naming pattern for the computed indexes
            columns. Must include "{name}" which will be replaced by the
            dimension name.
        method: "nearest" or "linear".
        cube_asserted: If False, *cube* will be verified, otherwise it
            is expected to be a valid cube.

    Returns:
        A new data frame whose columns are values from *cube* variables
        at given *indexes*.
    """
    if not cube_asserted:
        assert_cube(cube)

    if method not in {POINT_INTERP_METHOD_NEAREST, POINT_INTERP_METHOD_LINEAR}:
        raise ValueError(f"invalid method {method!r}")
    if method != POINT_INTERP_METHOD_NEAREST:
        raise NotImplementedError(f"method {method!r} not yet implemented")

    all_data_var_names = tuple(cube.data_vars.keys())
    if len(all_data_var_names) == 0:
        raise ValueError("cube is empty")

    if data_var_names is not None:
        if len(data_var_names) == 0:
            return _empty_dataset_from_points(indexes)
        for var_name in data_var_names:
            if var_name not in cube.data_vars:
                raise ValueError(f"variable {var_name!r} not found in cube")
    else:
        data_var_names = all_data_var_names

    dim_names = cube[data_var_names[0]].dims
    num_dims = len(dim_names)
    index_names = [index_name_pattern.format(name=dim_name) for dim_name in dim_names]
    indexes, num_points = _normalize_series(
        indexes, index_names, force_dataset=True, param_name="indexes"
    )
    if num_points == 0:
        return _empty_dataset_from_points(indexes)
    cube = xr.Dataset(
        {var_name: cube[var_name] for var_name in data_var_names}, coords=cube.coords
    )

    new_bounds_vars = {}
    bounds_var_names = _get_coord_bounds_var_names(cube)
    drop_coords = None
    if bounds_var_names:
        if include_bounds:
            # Flatten any coordinate bounds variables
            for var_name, bnds_var_name in bounds_var_names.items():
                bnds_var = cube[bnds_var_name]
                new_bounds_vars[f"{var_name}_lower"] = bnds_var[:, 0]
                new_bounds_vars[f"{var_name}_upper"] = bnds_var[:, 1]
            cube = cube.assign_coords(**new_bounds_vars)
        cube = cube.drop_vars(bounds_var_names.values())
        if not include_coords:
            drop_coords = set(cube.coords).difference(new_bounds_vars.keys())
    else:
        if not include_coords:
            drop_coords = set(cube.coords)

    # Generate a validation condition so we can
    # filter out invalid rows (where any index == -1)
    is_valid_point = None
    for index_name in index_names:
        col = indexes[index_name]
        condition = col >= 0 if np.issubdtype(col.dtype, np.integer) else np.isnan(col)
        if is_valid_point is None:
            is_valid_point = condition
        else:
            is_valid_point = np.logical_and(is_valid_point, condition)

    num_valid_points = np.count_nonzero(is_valid_point)
    if num_valid_points == num_points:
        # All indexes valid
        cube_selector = {dim_names[i]: indexes[index_names[i]] for i in range(num_dims)}
        cube_values = cube.isel(cube_selector)
    elif num_valid_points == 0:
        # All indexes are invalid
        new_bounds_vars = {}
        for var_name in cube.variables:
            new_bounds_vars[var_name] = _empty_points_var(cube[var_name], num_points)
        cube_values = xr.Dataset(new_bounds_vars)
    else:
        # Some invalid indexes
        idx = np.arange(num_points)
        good_idx = idx[is_valid_point.values]
        idx_dim_name = indexes[index_names[0]].dims[0]
        good_indexes = indexes.isel({idx_dim_name: good_idx})

        cube_selector = {
            dim_names[i]: good_indexes[index_names[i]] for i in range(num_dims)
        }
        cube_values = cube.isel(cube_selector)

        new_bounds_vars = {}
        for var_name in cube.variables:
            var = cube_values[var_name]
            new_var = _empty_points_var(var, num_points)
            new_var[good_idx] = var
            new_bounds_vars[var_name] = new_var

        cube_values = xr.Dataset(new_bounds_vars)

    if drop_coords:
        cube_values = cube_values.drop_vars(drop_coords)

    return cube_values


def get_cube_point_indexes(
    cube: xr.Dataset,
    points: PointsLike,
    dim_name_mapping: Mapping[str, str] = None,
    index_name_pattern: str = DEFAULT_INDEX_NAME_PATTERN,
    index_dtype=np.float64,
    cube_asserted: bool = False,
) -> xr.Dataset:
    """Get indexes of given point coordinates *points* into the given *dataset*.

    Args:
        cube: The cube dataset.
        points: A mapping from column names to column data arrays, which
            must all have the same length.
        dim_name_mapping: A mapping from dimension names in *cube* to
            column names in *points*.
        index_name_pattern: A naming pattern for the computed indexes
            columns. Must include "{name}" which will be replaced by the
            dimension name.
        index_dtype: Numpy data type for the indexes. If it is a
            floating point type (default), then *indexes* will contain
            fractions, which may be used for interpolation. For out-of-
            range coordinates in *points*, indexes will be -1 if
            *index_dtype* is an integer type, and NaN, if *index_dtype*
            is a floating point types.
        cube_asserted: If False, *cube* will be verified, otherwise it
            is expected to be a valid cube.

    Returns:
        A dataset containing the index columns.
    """
    if not cube_asserted:
        assert_cube(cube)

    dim_name_mapping = dim_name_mapping if dim_name_mapping is not None else {}
    dim_names = _get_cube_data_var_dims(cube)
    col_names = [
        dim_name_mapping.get(str(dim_name), dim_name) for dim_name in dim_names
    ]

    points, _ = _normalize_series(
        points, col_names, force_dataset=False, param_name="points"
    )

    indexes = []
    for dim_name, col_name in zip(dim_names, col_names):
        col = points[col_name]
        coord_indexes = get_dataset_indexes(
            cube, str(dim_name), col, index_dtype=index_dtype
        )
        indexes.append(
            (
                index_name_pattern.format(name=dim_name),
                xr.DataArray(coord_indexes, dims=[INDEX_DIM_NAME]),
            )
        )

    return xr.Dataset(dict(indexes))


def get_dataset_indexes(
    dataset: xr.Dataset,
    coord_var_name: str,
    coord_values: SeriesLike,
    index_dtype=np.float64,
) -> Union[xr.DataArray, np.ndarray]:
    """Compute the indexes and their fractions into a coordinate variable
    *coord_var_name* of a *dataset* for the given coordinate values
    *coord_values*.

    The coordinate variable's labels must be monotonic increasing or
    decreasing, otherwise the result will be nonsense.

    For any value in *coord_values* that is out of the bounds of the
    coordinate variable's values, the index depends on the value of
    *index_dtype*. If *index_dtype* is an integer type, invalid indexes
    are encoded as -1 while for floating point types, NaN will be used.

    Returns a tuple of indexes as int64 array and fractions as
    float64 array.

    Args:
        dataset: A cube dataset.
        coord_var_name: Name of a coordinate variable.
        coord_values: Array-like coordinate values.
        index_dtype: Numpy data type for the indexes. If it is floating
            point type (default), then *indexes* contain fractions,
            which may be used for interpolation. If *dtype* is an
            integer type out-of-range coordinates are indicated by index
            -1, and NaN if it is is a floating point type.

    Returns:
        The indexes and their fractions as a tuple of numpy int64 and
        float64 arrays.
    """
    coord_var = dataset[coord_var_name]
    n1 = coord_var.size
    n2 = n1 + 1

    coord_bounds_var = _get_bounds_var(dataset, coord_var_name)
    if coord_bounds_var is not None:
        coord_bounds = coord_bounds_var.values
        if np.issubdtype(coord_bounds.dtype, np.datetime64):
            coord_bounds = coord_bounds.astype(np.uint64)
        is_inverse = (coord_bounds[0, 1] - coord_bounds[0, 0]) < 0
        if is_inverse:
            coord_bounds = coord_bounds[::-1, ::-1]
        coords = np.zeros(n2, dtype=coord_bounds.dtype)
        coords[0:-1] = coord_bounds[:, 0]
        coords[-1] = coord_bounds[-1, 1]
    elif coord_var.size > 1:
        center_coords = coord_var.values
        if np.issubdtype(center_coords.dtype, np.datetime64):
            center_coords = center_coords.astype(np.uint64)
        is_inverse = (center_coords[-1] - center_coords[0]) < 0
        if is_inverse:
            center_coords = center_coords[::-1]
        deltas = np.zeros(n2, dtype=center_coords.dtype)
        deltas[0:-2] = np.diff(center_coords)
        deltas[-2] = deltas[-3]
        deltas[-1] = deltas[-3]
        coords = np.zeros(n2, dtype=center_coords.dtype)
        coords[0:-1] = center_coords
        coords[-1] = coords[-2] + deltas[-1]
        if np.issubdtype(deltas.dtype, np.integer):
            coords -= deltas // 2
        else:
            coords -= 0.5 * deltas
    else:
        raise ValueError(
            f"cannot determine cell boundaries for"
            f" coordinate variable {coord_var_name!r}"
            f" of size {coord_var.size}"
        )

    if np.issubdtype(coord_values.dtype, np.datetime64):
        try:
            coord_values = coord_values.astype(np.uint64)
        except TypeError:
            # Fixes https://github.com/dcs4cop/xcube/issues/95
            coord_values = coord_values.values.astype(np.uint64)

    indexes = np.linspace(0.0, n1, n2, dtype=np.float64)
    interp_indexes = np.interp(coord_values, coords, indexes, left=-1, right=-1)
    if is_inverse:
        i = interp_indexes >= 0
        interp_indexes[i] = n1 - interp_indexes[i]
    interp_indexes[interp_indexes >= n1] = n1 - 1e-9
    if np.issubdtype(index_dtype, np.integer):
        interp_indexes = interp_indexes.astype(index_dtype)
    else:
        interp_indexes[interp_indexes < 0] = np.nan

    return interp_indexes


def _empty_points_var(var: xr.DataArray, num_points: int):
    fill_value = 0 if np.issubdtype(var.dtype, np.integer) else np.nan
    return xr.DataArray(
        np.full(num_points, fill_value, dtype=var.dtype),
        dims=[INDEX_DIM_NAME],
        attrs=var.attrs,
    )


def _normalize_series(
    points: PointsLike,
    col_names: Sequence[str],
    force_dataset: bool = False,
    param_name: str = "points",
) -> tuple[PointsLike, int]:
    if not isinstance(points, (xr.Dataset, pd.DataFrame)):
        new_points = {}
        for col_name in col_names:
            if col_name in points:
                col = points[col_name]
                if not isinstance(col, xr.DataArray):
                    col = xr.DataArray(col)
                new_points[col_name] = col
        points = new_points

    num_points = None
    for col_name in col_names:
        if col_name not in points:
            raise ValueError(f"column {col_name!r} not found in {param_name}")
        col = points[col_name]
        if len(col.shape) != 1:
            raise ValueError(
                f"column {col_name!r} in {param_name} must be"
                f" one-dimensional, but has shape {col.shape!r}"
            )
        if num_points is None:
            num_points = len(col)
        elif num_points != len(col):
            raise ValueError(
                f"column sizes in {param_name} must be all the"
                f" same, but found {len(points[col_names[0]])}"
                f" for column {col_names[0]!r} and {num_points}"
                f" for column {col_name!r}"
            )

    if num_points is None:
        raise ValueError(f"{param_name} has no valid columns")

    if force_dataset:
        if isinstance(points, pd.DataFrame):
            points = xr.Dataset.from_dataframe(points)
        elif not isinstance(points, xr.DataArray):
            points = xr.Dataset(points)

    return points, num_points


def _get_coord_bounds_var_names(dataset: xr.Dataset) -> dict[Hashable, Any]:
    bnds_var_names = {}
    for var_name in dataset.coords:
        var = dataset[var_name]
        bnds_var_name = var.attrs.get("bounds", f"{var_name}_bnds")
        if bnds_var_name not in bnds_var_names and bnds_var_name in dataset:
            bnds_var = dataset[bnds_var_name]
            if bnds_var.shape == (var.shape[0], 2):
                bnds_var_names[var_name] = bnds_var_name
    return bnds_var_names


def _get_cube_data_var_dims(cube: xr.Dataset) -> tuple[Hashable, ...]:
    for var in cube.data_vars.values():
        return var.dims
    raise ValueError("cube dataset is empty")


def _get_bounds_var(dataset: xr.Dataset, var_name: str) -> Optional[xr.DataArray]:
    var = dataset[var_name]
    if len(var.shape) == 1:
        bounds_var_name = var.attrs.get("bounds", f"{var_name}_bnds")
        if bounds_var_name in dataset:
            bounds_var = dataset[bounds_var_name]
            if bounds_var.dtype == var.dtype and bounds_var.shape == (var.size, 2):
                return bounds_var
    return None


def _empty_dataset_from_points(data: Any) -> xr.Dataset:
    return xr.Dataset(coords=data.coords if hasattr(data, "coords") else None)
