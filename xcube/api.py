from contextlib import contextmanager
from typing import Dict, List, Union, Tuple, Any, Optional, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from xcube.dsio import find_dataset_io, guess_dataset_format, FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4

_TIME_DTYPE = "datetime64[s]"
_TIME_UNITS = "seconds since 1970-01-01T00:00:00"
_TIME_CALENDAR = "proleptic_gregorian"

INDEX_NAME_PATTERN = '{name}_index'
FRACTION_NAME_PATTERN = '{name}_fraction'
INDEX_DIM_NAME = "idx"


def new_cube(title="Test Cube",
             width=360,
             height=180,
             spatial_res=1.0,
             lon_start=-180.0,
             lat_start=-90.0,
             time_periods=5,
             time_freq="D",
             time_start='2010-01-01T00:00:00',
             drop_bounds=False,
             variables=None):
    """
    Create a new empty cube. Useful for testing.

    :param title: A title.
    :param width: Horizontal number of grid cells
    :param height: Vertical number of grid cells
    :param spatial_res: Spatial resolution in degrees
    :param lon_start: Minimum longitude value
    :param lat_start: Minimum latitude value
    :param time_periods: Number of time steps
    :param time_freq: Duration of each time step
    :param time_start: First time value
    :param drop_bounds: If True, coordinate bounds variables are not created.
    :param variables: Dictionary of data variables to be added.
    :return: A cube instance
    """
    lon_end = lon_start + width * spatial_res
    lat_end = lat_start + height * spatial_res
    if width < 0 or height < 0 or spatial_res <= 0.0:
        raise ValueError()
    if lon_start < -180. or lon_end > 180. or lat_start < -90. or lat_end > 90.:
        raise ValueError()
    if time_periods < 0:
        raise ValueError()

    lon_data = np.linspace(lon_start + 0.5 * spatial_res, lon_end - 0.5 * spatial_res, width)
    lon = xr.DataArray(lon_data, dims="lon")
    lon.attrs["units"] = "degrees_east"

    lat_data = np.linspace(lat_start + 0.5 * spatial_res, lat_end - 0.5 * spatial_res, height)
    lat = xr.DataArray(lat_data, dims="lat")
    lat.attrs["units"] = "degrees_north"

    time_data_2 = pd.date_range(start=time_start, periods=time_periods + 1, freq=time_freq).values
    time_data_2 = time_data_2.astype(dtype=_TIME_DTYPE)
    time_delta = time_data_2[1] - time_data_2[0]
    time_data = time_data_2[0:-1] + time_delta // 2
    time = xr.DataArray(time_data, dims="time")
    time.encoding["units"] = _TIME_UNITS
    time.encoding["calendar"] = _TIME_CALENDAR

    time_data_2 = pd.date_range(time_start, periods=time_periods + 1, freq=time_freq)

    coords = dict(lon=lon, lat=lat, time=time)
    if not drop_bounds:
        lon_bnds_data = np.zeros((width, 2), dtype=np.float64)
        lon_bnds_data[:, 0] = np.linspace(lon_start, lon_end - spatial_res, width)
        lon_bnds_data[:, 1] = np.linspace(lon_start + spatial_res, lon_end, width)
        lon_bnds = xr.DataArray(lon_bnds_data, dims=("lon", "bnds"))
        lon_bnds.attrs["units"] = "degrees_east"

        lat_bnds_data = np.zeros((height, 2), dtype=np.float64)
        lat_bnds_data[:, 0] = np.linspace(lat_start, lat_end - spatial_res, height)
        lat_bnds_data[:, 1] = np.linspace(lat_start + spatial_res, lat_end, height)
        lat_bnds = xr.DataArray(lat_bnds_data, dims=("lat", "bnds"))
        lat_bnds.attrs["units"] = "degrees_north"

        time_bnds_data = np.zeros((time_periods, 2), dtype="datetime64[ns]")
        time_bnds_data[:, 0] = time_data_2[:-1]
        time_bnds_data[:, 1] = time_data_2[1:]
        time_bnds = xr.DataArray(time_bnds_data, dims=("time", "bnds"))
        time_bnds.encoding["units"] = _TIME_UNITS
        time_bnds.encoding["calendar"] = _TIME_CALENDAR

        lon.attrs["bounds"] = "lon_bnds"
        lat.attrs["bounds"] = "lat_bnds"
        time.attrs["bounds"] = "time_bnds"

        coords.update(dict(lon_bnds=lon_bnds, lat_bnds=lat_bnds, time_bnds=time_bnds))

    attrs = {
        "Conventions": "CF-1.7",
        "title": title,
        "time_coverage_start": str(time_data_2[0]),
        "time_coverage_end": str(time_data_2[-1]),
        "geospatial_lon_min": lon_start,
        "geospatial_lon_max": lon_end,
        "geospatial_lon_units": "degrees_east",
        "geospatial_lat_min": lat_start,
        "geospatial_lat_max": lat_end,
        "geospatial_lat_units": "degrees_north",
    }

    data_vars = {}
    if variables:
        dims = ("time", "lat", "lon")
        shape = (time_periods, height, width)
        size = time_periods * height * width
        for var_name, data in variables.items():
            if isinstance(data, xr.DataArray):
                data_vars[var_name] = data
            elif isinstance(data, int) or isinstance(data, float) or isinstance(data, bool):
                data_vars[var_name] = xr.DataArray(np.full(shape, data), dims=dims)
            elif data is None:
                data_vars[var_name] = xr.DataArray(np.random.uniform(0.0, 1.0, size).reshape(shape), dims=dims)
            else:
                data_vars[var_name] = xr.DataArray(data, dims=dims)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


@contextmanager
def open_dataset(input_path: str,
                 format_name: str = None,
                 is_cube: bool = False,
                 **kwargs) -> xr.Dataset:
    """
    The ``read_dataset`` function as context manager that auto-closes the dataset read.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param is_cube: Weather a ValueError will be raised, if the dataset read from *input_path* is not a data cube.
    :param kwargs: format-specific keyword arguments
    :return: dataset object
    """
    dataset = read_dataset(input_path, format_name, **kwargs)
    if is_cube:
        assert_cube(dataset)
    try:
        yield dataset
    finally:
        dataset.close()


def read_dataset(input_path: str,
                 format_name: str = None,
                 **kwargs) -> xr.Dataset:
    """
    Read a dataset.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: dataset object
    """
    format_name = format_name if format_name else guess_dataset_format(input_path)
    if format_name is None:
        raise ValueError("Unknown input format")
    dataset_io = find_dataset_io(format_name, modes=["r"])
    if dataset_io is None:
        raise ValueError(f"Unknown input format {format_name!r} for {input_path}")

    return dataset_io.read(input_path, **kwargs)


def write_dataset(dataset: xr.Dataset,
                  output_path: str,
                  format_name: str = None,
                  **kwargs) -> xr.Dataset:
    """
    Write dataset.

    :param dataset: Dataset to be written.
    :param output_path: output path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: the input dataset
    """
    format_name = format_name if format_name else guess_dataset_format(output_path)
    if format_name is None:
        raise ValueError("Unknown output format")
    dataset_io = find_dataset_io(format_name, modes=["w"])
    if dataset_io is None:
        raise ValueError(f"Unknown output format {format_name!r} for {output_path}")

    dataset_io.write(dataset, output_path, **kwargs)

    return dataset


def chunk_dataset(dataset: xr.Dataset,
                  chunk_sizes: Dict[str, int] = None,
                  format_name: str = None) -> xr.Dataset:
    """
    Chunk dataset and update encodings for given format.

    :param dataset: input dataset
    :param chunk_sizes: mapping from dimension name to new chunk size
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :return: the re-chunked dataset
    """
    chunked_ds = dataset.chunk(chunks=chunk_sizes)

    # Update encoding so writing of chunked_ds recognizes new chunks
    chunk_sizes_attr_name = None
    if format_name == FORMAT_NAME_ZARR:
        chunk_sizes_attr_name = "chunks"
    if format_name == FORMAT_NAME_NETCDF4:
        chunk_sizes_attr_name = "chunksizes"
    if chunk_sizes_attr_name:
        for var_name in chunked_ds.variables:
            var = chunked_ds[var_name]
            if chunk_sizes:
                sizes = tuple(chunk_sizes[dim_name] if dim_name in chunk_sizes
                              else var.shape[var.dims.index(dim_name)]
                              for dim_name in var.dims)
                var.encoding.update({chunk_sizes_attr_name: sizes})
            elif chunk_sizes_attr_name in var.encoding:
                # Remove any explicit and wrong specification so writing will use Dask chunks (TBC!)
                del var.encoding[chunk_sizes_attr_name]

    return chunked_ds


def dump_dataset(dataset: xr.Dataset,
                 variable_names=None,
                 show_var_encoding=False) -> str:
    """
    Dumps a dataset or variables into a text string.

    :param dataset: input dataset
    :param variable_names: names of variables to be dumped
    :param show_var_encoding: also dump variable encodings?
    :return: the dataset dump
    """
    lines = []
    if not variable_names:
        lines.append(str(dataset))
        if show_var_encoding:
            for var_name, var in dataset.coords.items():
                if var.encoding:
                    lines.append(dump_var_encoding(var, header=f"Encoding for coordinate variable {var_name!r}:"))
            for var_name, var in dataset.data_vars.items():
                if var.encoding:
                    lines.append(dump_var_encoding(var, header=f"Encoding for data variable {var_name!r}:"))
    else:
        for var_name in variable_names:
            var = dataset[var_name]
            lines.append(str(var))
            if show_var_encoding and var.encoding:
                lines.append(dump_var_encoding(var))
    return "\n".join(lines)


def dump_var_encoding(var: xr.DataArray, header="Encoding:", indent=4) -> str:
    """
    Dump the encoding information of a variable into a text string.

    :param var: Dataset variable.
    :param header: Title/header string.
    :param indent: Indention in spaces.
    :return: the variable dump
    """
    lines = [header]
    max_len = 0
    for k in var.encoding:
        max_len = max(max_len, len(k))
    indent_spaces = indent * " "
    for k, v in var.encoding.items():
        tab_spaces = (2 + max_len - len(k)) * " "
        lines.append(f"{indent_spaces}{k}:{tab_spaces}{v!r}")
    return "\n".join(lines)


def get_cube_values_for_points(cube: xr.Dataset,
                               points: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
                               var_names: Sequence[str] = None,
                               index_name_pattern: str = INDEX_NAME_PATTERN,
                               include_indexes: bool = False,
                               method: str = "nearest",
                               cube_asserted: bool = False) -> xr.Dataset:
    """
    Extract values from *cube* variables at given coordinates in *points*.

    :param cube: The cube dataset.
    :param points: Dictionary that maps dimension name to coordinate arrays.
    :param var_names: An optional list of names of data variables in *cube* whose values shall be extracted.
    :param index_name_pattern: A naming pattern for the computed indexes columns.
           Must include "{name}" which will be replaced by the dimension name.
    :param include_indexes: Weather to include computed indexes in return value.
    :param method: "nearest" or "linear".
    :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
    :return: A new data frame whose columns are values from *cube* variables at given *points*.
    """
    if not cube_asserted:
        assert_cube(cube)

    point_indexes = get_cube_point_indexes(cube,
                                           points,
                                           index_name_pattern=index_name_pattern,
                                           dtype=np.int64 if method == "nearest" else np.float64,
                                           cube_asserted=True)

    cube_values = get_cube_values_for_indexes(cube,
                                              point_indexes,
                                              var_names=var_names,
                                              index_name_pattern=index_name_pattern,
                                              method=method,
                                              cube_asserted=True)

    if include_indexes:
        cube_values.update(point_indexes, inplace=True)

    return cube_values


def get_cube_values_for_indexes(cube: xr.Dataset,
                                indexes: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
                                var_names: Sequence[str] = None,
                                index_name_pattern: str = INDEX_NAME_PATTERN,
                                method: str = "nearest",
                                cube_asserted: bool = False) -> xr.Dataset:
    """
    Get values from the *cube* at given *indexes*.

    :param cube: A cube dataset.
    :param indexes: A mapping from column names to index and fraction arrays for all cube dimensions.
    :param var_names: An optional list of names of data variables in *cube* whose values shall be extracted.
    :param index_name_pattern: A naming pattern for the computed indexes columns.
           Must include "{name}" which will be replaced by the dimension name.
    :param method: "nearest" or "linear".
    :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
    :return: A new data frame whose columns are values from *cube* variables at given *indexes*.
    """
    if not cube_asserted:
        assert_cube(cube)

    if method not in {"nearest", "linear"}:
        raise ValueError(f"invalid method {method!r}")
    if method != "nearest":
        raise NotImplementedError(f"method {method!r} not yet implemented")

    all_var_names = tuple(cube.data_vars.keys())
    if len(all_var_names) == 0:
        raise ValueError("cube is empty")

    if var_names is not None:
        if len(var_names) == 0:
            return xr.Dataset(coords=indexes.coords if hasattr(indexes, "coords") else None)
        for var_name in var_names:
            if var_name not in cube.data_vars:
                raise ValueError(f"variable {var_name!r} not found in cube")
    else:
        var_names = all_var_names

    dim_names = cube[var_names[0]].dims
    num_dims = len(dim_names)
    index_names = [index_name_pattern.format(name=dim_name) for dim_name in dim_names]
    num_points = _validate_points(indexes, index_names, param_name="indexes")
    indexes = _normalize_points(indexes)

    # Flatten any coordinate bounds variables
    cube = xr.Dataset({var_name: cube[var_name] for var_name in var_names}, coords=cube.coords)
    bounds_var_names = _get_coord_bounds_var_names(cube)
    new_vars = {}
    for var_name, bnds_var_name in bounds_var_names.items():
        bnds_var = cube[bnds_var_name]
        new_vars[var_name + "_start"] = bnds_var[:, 0]
        new_vars[var_name + "_end"] = bnds_var[:, 1]
    cube = cube.drop(list(bounds_var_names.keys()))
    cube = cube.assign_coords(**new_vars)

    # Generate a validation condition so we can filter out invalid rows (where any index == -1)
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
        new_vars = {}
        for var_name in var_names:
            new_vars[var_name] = _empty_points_var(cube[var_name], num_points)
        cube_values = xr.Dataset(new_vars)
    else:
        # Some invalid indexes
        idx = np.arange(num_points)
        good_idx = idx[is_valid_point.values]
        idx_dim_name = indexes[index_names[0]].dims[0]
        good_indexes = indexes.isel({idx_dim_name: good_idx})

        cube_selector = {dim_names[i]: good_indexes[index_names[i]] for i in range(num_dims)}
        cube_values = cube.isel(cube_selector)

        new_vars = {}
        for var_name in var_names:
            var = cube_values[var_name]
            new_var = _empty_points_var(var, num_points)
            new_var[good_idx] = var
            new_vars[var_name] = new_var

        cube_values = xr.Dataset(new_vars)

    return cube_values


def _empty_points_var(var: xr.DataArray, num_points):
    fill_value = 0 if np.issubdtype(var.dtype, np.integer) else np.nan
    return xr.DataArray(np.full(num_points, fill_value, dtype=var.dtype),
                        dims=[INDEX_DIM_NAME], attrs=var.attrs)


def _normalize_points(points: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]]) -> xr.Dataset:
    if isinstance(points, pd.DataFrame):
        points = xr.Dataset.from_dataframe(points)
    elif not isinstance(points, xr.DataArray):
        points = xr.Dataset(points)
    return points


def _validate_points(points: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
                     col_names: Sequence[str],
                     param_name="points") -> Optional[int]:
    num_points = None
    for col_name in col_names:
        if col_name not in points:
            raise ValueError(f"column {col_name!r} not found in {param_name}")
        col = points[col_name]
        if len(col.shape) != 1:
            raise ValueError(
                f"column {col_name!r} in {param_name} must be one-dimensional, but has shape {col.shape!r}")
        if num_points is None:
            num_points = len(col)
        elif num_points != len(col):
            raise ValueError(f"column sizes in {param_name} must be all the same,"
                             f" but found {len(points[col_names[0]])} for column {col_names[0]!r}"
                             f" and {num_points} for column {col_name!r}")
    return num_points


def get_cube_point_indexes(cube: xr.Dataset,
                           points: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
                           dim_name_mapping: Mapping[str, str] = None,
                           index_name_pattern: str = INDEX_NAME_PATTERN,
                           dtype=np.float64,
                           cube_asserted: bool = False) -> xr.Dataset:
    """
    Get indexes of given point coordinates *points* into the given *dataset*.

    :param cube: The cube dataset.
    :param points: A mapping from column names to column data arrays, which must all have the same length.
    :param dim_name_mapping: A mapping from dimension names in *cube* to column names in *points*.
    :param index_name_pattern: A naming pattern for the computed indexes columns.
           Must include "{name}" which will be replaced by the dimension name.
    :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
    :param dtype: Numpy data type for the indexes. If it is floating point type (default),
           then *indexes* contain fractions, which may be used for interpolation. If *dtype* is an integer
           type out-of-range coordinates are indicated by index -1, and NaN if it is is a floating point type.
    :return: A dataset containing the index columns.
    """
    if not cube_asserted:
        assert_cube(cube)

    dim_name_mapping = dim_name_mapping if dim_name_mapping is not None else {}
    dim_names = _get_cube_data_var_dims(cube)
    col_names = [dim_name_mapping.get(dim_name, dim_name) for dim_name in dim_names]

    _validate_points(points, col_names, param_name="points")

    indexes = []
    for dim_name, col_name in zip(dim_names, col_names):
        col = points[col_name]
        coord_indexes = get_dataset_indexes(cube, dim_name, col, dtype=dtype)
        indexes.append((index_name_pattern.format(name=dim_name),
                        xr.DataArray(coord_indexes, dims=[INDEX_DIM_NAME])))

    return xr.Dataset(dict(indexes))


def _get_coord_bounds_var_names(dataset: xr.Dataset) -> Dict[str, str]:
    bnds_var_names = {}
    for var_name in dataset.coords:
        var = dataset[var_name]
        bnds_var_name = var.attrs.get("bounds", var_name + "_bnds")
        if bnds_var_name not in bnds_var_names and bnds_var_name in dataset:
            bnds_var = dataset[bnds_var_name]
            if bnds_var.shape == (var.shape[0], 2):
                bnds_var_names[var_name] = bnds_var_name
    return bnds_var_names


def _get_cube_data_var_dims(cube: xr.Dataset) -> Tuple[str, ...]:
    for var in cube.data_vars.values():
        return var.dims
    raise ValueError("cube dataset is empty")


def get_dataset_indexes(dataset: xr.Dataset,
                        coord_var_name: str,
                        coord_values: Union[xr.DataArray, np.ndarray],
                        dtype=np.float64) -> Union[xr.DataArray, np.ndarray]:
    """
    Compute the indexes and their fractions into a coordinate variable *coord_var_name* of a *dataset*
    for the given coordinate values *coord_values*.

    The coordinate variable's labels must be monotonic increasing or decreasing,
    otherwise the result will be nonsense.

    For any value in *coord_values* that is out of the bounds of the coordinate variable's values,
    the index will be -1 and the fraction will be NaN.

    Returns a tuple of indexes as int64 array and fractions as float64 array.

    :param dataset: A cube dataset.
    :param coord_var_name: Name of a coordinate variable.
    :param coord_values: Array-like coordinate values.
    :param dtype: Numpy data type for the indexes. If it is floating point type (default),
           then *indexes* contain fractions, which may be used for interpolation. If *dtype* is an integer
           type out-of-range coordinates are indicated by index -1, and NaN if it is is a floating point type.
    :return: The indexes and their fractions as a tuple of numpy int64 and float64 arrays.
    """
    coord_var = dataset[coord_var_name]
    n1 = coord_var.size
    n2 = n1 + 1

    coord_bounds_var = _get_bounds_var(dataset, coord_var_name)
    if coord_bounds_var is not None:
        coord_bounds = coord_bounds_var.values
        if np.issubdtype(coord_bounds.dtype, np.datetime64):
            coord_bounds = coord_bounds.astype(np.uint64)
        is_reversed = (coord_bounds[0, 1] - coord_bounds[0, 0]) < 0
        if is_reversed:
            coord_bounds = coord_bounds[::-1, ::-1]
        coords = np.zeros(n2, dtype=coord_bounds.dtype)
        coords[0:-1] = coord_bounds[:, 0]
        coords[-1] = coord_bounds[-1, 1]
    elif coord_var.size > 1:
        center_coords = coord_var.values
        if np.issubdtype(center_coords.dtype, np.datetime64):
            center_coords = center_coords.astype(np.uint64)
        is_reversed = (center_coords[-1] - center_coords[0]) < 0
        if is_reversed:
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
        raise ValueError(f"cannot determine cell boundaries for"
                         f" coordinate variable {coord_var_name!r} of size {coord_var.size}")

    if np.issubdtype(coord_values.dtype, np.datetime64):
        coord_values = coord_values.astype(np.uint64)

    indexes = np.linspace(0.0, n1, n2, dtype=np.float64)
    interp_indexes = np.interp(coord_values, coords, indexes, left=-1, right=-1)
    if np.issubdtype(dtype, np.integer):
        interp_indexes = interp_indexes.astype(dtype)
        interp_indexes[interp_indexes >= n1] = n1 - 1
    else:
        interp_indexes[interp_indexes >= n1] = n1 - 1e-9
        interp_indexes[interp_indexes < 0] = np.nan

    if is_reversed:
        interp_indexes = interp_indexes[::-1]

    return interp_indexes


def get_cube_chunks(cube: xr.Dataset) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """
    Get chunk sizes of the given *cube* dataset.

    The function returns the chunks of the first data variable, if any,
    because all data variables in a cube are expected to have same chunk sizes.

    :param cube: A cube dataset
    :return: A tuple of cube sizes for each cube dimension.
    """
    for var_name in cube.data_vars:
        var_data = cube[var_name].data
        return var_data.chunks if hasattr(var_data, "chunks") else None
    return None


def get_cube_dims(cube: xr.Dataset) -> Optional[Tuple[str]]:
    """
    Get dimension names of the given *cube* dataset.

    The function returns the dimensions of the first data variable, if any,
    because all data variables in a cube are expected to have same dimensions.

    :param cube: A cube dataset
    :return: A tuple of dimension names or None if the cube has no data variables.
    """
    for var_name in cube.data_vars:
        return cube[var_name].dims
    return None


def _get_bounds_var(dataset: xr.Dataset, var_name: str) -> Optional[xr.DataArray]:
    var = dataset[var_name]
    if len(var.shape) == 1:
        bounds_var_name = var.attrs.get("bounds", var_name + "_bnds")
        if bounds_var_name in dataset:
            bounds_var = dataset[bounds_var_name]
            if bounds_var.dtype == var.dtype and bounds_var.shape == (var.size, 2):
                return bounds_var
    return None


def assert_cube(dataset: xr.Dataset, name=None):
    """
    Assert that the given *dataset* is a valid data cube.

    :param dataset: The dataset to be validated.
    :param name: Optional parameter name.
    :raise: ValueError, if dataset is not a valid data cube
    """
    report = verify_cube(dataset)
    if report:
        message = f"Dataset" + (name + " " if name else " ")
        message += "is not a valid data cube, because:\n"
        message += "- " + ";\n- ".join(report) + "."
        raise ValueError(message)


def verify_cube(dataset: xr.Dataset) -> List[str]:
    """
    Verify the given *dataset* for being a valid data cube.

    Returns a list of issues, which is empty if *dataset* is a valid data cube.

    :param dataset: A dataset to be verified.
    :return: List of issues or empty list.
    """
    report = []
    _check_dim(dataset, "time", report)
    _check_dim(dataset, "lat", report)
    _check_dim(dataset, "lon", report)
    _check_time(dataset, "time", report)
    _check_lon_or_lat(dataset, "lat", -90, 90, report)
    _check_lon_or_lat(dataset, "lon", -180, 180, report)
    _check_data_variables(dataset, report)
    return report


def _check_data_variables(dataset, report):
    first_var = None
    first_dims = None
    first_chunks = None
    for var_name, var in dataset.data_vars.items():
        dims = var.dims
        chunks = var.data.chunks if hasattr(var.data, "chunks") else None

        if len(dims) < 3 or dims[0] != "time" or dims[-2] != "lat" or dims[-1] != "lon":
            report.append(f"dimensions of data variable {var_name!r}"
                          f" must be ('time', ..., 'lat', 'lon'),"
                          f" but were {dims!r} for {var_name!r}")

        if first_var is None:
            first_var = var
            first_dims = dims
            first_chunks = chunks
            continue

        if first_dims != dims:
            report.append("dimensions of all data variables must be same,"
                          f" but found {first_dims!r} for {first_var.name!r} "
                          f"and {dims!r} for {var_name!r}")

        if first_chunks != chunks:
            report.append("all data variables must have same chunk sizes,"
                          f" but found {first_chunks!r} for {first_var.name!r} "
                          f"and {chunks!r} for {var_name!r}")


def _check_dim(dataset, name, report):
    if name not in dataset.dims:
        report.append(f"missing dimension {name!r}")

    if dataset.dims[name] < 0:
        report.append(f"size of dimension {name!r} must be a positive integer")


def _check_coord_var(dataset, name, report):
    if name not in dataset.coords:
        report.append(f"missing coordinate variable {name!r}")
        return None

    var = dataset.coords[name]
    if var.dims != (name,):
        report.append(f"coordinate variable {name!r} must have a single dimension {name!r}")
        return None

    if var.size == 0:
        report.append(f"coordinate variable {name!r} must not be empty")
        return None

    return var


def _check_lon_or_lat(dataset, name, min_value, max_value, report):
    var = _check_coord_var(dataset, name, report)
    if var is None:
        return

    if not np.all(np.isfinite(var)):
        report.append(f"values of coordinate variable {name!r} must be finite")

    if np.min(var) < min_value or np.max(var) > max_value:
        report.append(f"values of coordinate variable {name!r}"
                      f" must be in the range {min_value} to {max_value}")

    # TODO (forman): the following check is not valid for "lat" because we currently use wrong lat-order
    # TODO (forman): the following check is not valid for "lon" if a cube covers the antimeridian
    # if not np.all(np.diff(var.astype(np.float64)) > 0):
    #    report.append(f"values of coordinate variable {name!r} must be monotonic increasing")


def _check_time(dataset, name, report):
    var = _check_coord_var(dataset, name, report)
    if var is None:
        return

    if not np.issubdtype(var.dtype, np.datetime64):
        report.append(f"type of coordinate variable {name!r} must be datetime64")

    if not np.all(np.diff(var.astype(np.float64)) > 0):
        report.append(f"values of coordinate variable {name!r} must be monotonic increasing")
