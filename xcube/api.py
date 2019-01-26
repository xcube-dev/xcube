from contextlib import contextmanager
from typing import Dict, List, Union, Tuple, Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from xcube.dsio import find_dataset_io, guess_dataset_format, FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4


@contextmanager
def open_dataset(input_path: str,
                 format_name: str = None,
                 **kwargs) -> xr.Dataset:
    """
    The ``read_dataset`` function as context manager that auto-closes the dataset read.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: dataset object
    """
    dataset = read_dataset(input_path, format_name, **kwargs)
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


def get_cube_values(cube: xr.Dataset,
                    indexes: pd.DataFrame,
                    cube_asserted: bool = False):
    if not cube_asserted:
        assert_cube(cube)

    row_indexes = np.stack([indexes[c] for c in indexes], axis=-1)

    num_rows = len(row_indexes)

    vars = [cube[var_name] for var_name in cube.data_vars]
    var_names = [var_name for var_name in cube.data_vars]
    var_values = [np.ndarray((num_rows,), dtype=cube[var_name].dtype) for var_name in cube.data_vars]
    num_vars = len(var_names)

    for i in range(num_rows):
        row_index = tuple(row_indexes[i])
        if -1 in row_index:
            for j in range(num_vars):
                var_values[j][i] = np.nan
        else:
            for j in range(num_vars):
                var = vars[j]
                print(row_index, var)
                var_values[j][i] = var[row_index]

    return pd.DataFrame({var_names[j]: var_values[j] for j in range(num_vars)})


def get_cube_point_values(cube: xr.Dataset,
                          points: Union[pd.DataFrame, Any],
                          cube_asserted: bool = False):
    """
    Extract values for *points* from *cube*.

    :param cube: The cube dataset.
    :param points: Dictionary that maps dimension name to coordinate arrays.
    :param cube_asserted: If False, *cube* will be validated, otherwise it is expected to be a valid cube.
    :return:
    """
    if not cube_asserted:
        assert_cube(cube)
    indexes = get_cube_point_indexes(cube, points, cube_asserted=True)
    return get_cube_values(cube, indexes, cube_asserted=True)


def get_cube_point_indexes(cube: xr.Dataset,
                           points: Union[pd.DataFrame, Any],
                           dim_name_mapping: Dict[str, str] = None,
                           cube_asserted: bool = False) -> pd.DataFrame:
    """
    Get indexes of given point coordinates *points* into the given *dataset*.

    :param cube: The cube dataset.
    :param points: A pandas data frame or object that can be converted into a Pandas DataFrame.
    :param dim_name_mapping: A mapping from dimension names in *cube* to column names in *points*.
    :param cube_asserted: If False, *cube* will be validated, otherwise it is expected to be a valid cube.
    :return: A dictionary that maps dimension names to integer index arrays.
    """
    if not cube_asserted:
        assert_cube(cube)

    if not isinstance(points, pd.DataFrame):
        points = pd.DataFrame(points)

    dim_names = _get_cube_data_var_dims(cube)

    first_col_name = None
    indexes = []
    for dim_name in dim_names:
        if dim_name not in cube.coords:
            raise ValueError(f"missing coordinate variable for dimension {dim_name!r}")
        col_name = dim_name_mapping[dim_name] if dim_name_mapping and dim_name in dim_name_mapping else dim_name
        if col_name not in points:
            raise ValueError(f"column {col_name!r} not found in points")
        dim_col = points[col_name]
        if first_col_name is None:
            first_col_name = col_name
        else:
            first_col_size = len(points[first_col_name])
            col_size = len(dim_col)
            if first_col_size != col_size:
                raise ValueError("number of point coordinates must be same for all columns,"
                                 f" but found {first_col_size} for column {first_col_name!r}"
                                 f" and {col_size} for column {col_name!r}")

        indexes.append((dim_name, _get_coord_index(cube, dim_name, dim_col.values)))

    return pd.DataFrame(dict(indexes))


def _get_cube_data_var_dims(cube: xr.Dataset) -> Tuple[str, ...]:
    for var in cube.data_vars.values():
        return var.dims
    raise ValueError("cube dataset is empty")


def _get_coord_index(cube: xr.Dataset,
                     coord_var_name: str,
                     coord_value, dtype=np.int64) -> np.ndarray:
    coord_var = cube[coord_var_name]
    n1 = coord_var.size
    n2 = n1 + 1

    coord_bounds_var = _get_bounds_var(cube, coord_var_name)
    if coord_bounds_var is not None:
        coords = np.zeros(n2, dtype=coord_var.dtype)
        coords[0:-2] = coord_bounds_var[:, 0]
        coords[-1] = coord_bounds_var[-1, 1]
        if np.issubdtype(coords.dtype, np.datetime64):
            coords = coords.astype(np.uint64)
    elif coord_var.size > 1:
        center_coords = coord_var.values
        if np.issubdtype(center_coords.dtype, np.datetime64):
            center_coords = center_coords.astype(np.uint64)
        deltas = np.zeros(n2, dtype=center_coords.dtype)
        deltas[0:-2] = np.diff(center_coords)
        deltas[-2] = deltas[-3]
        deltas[-1] = deltas[-3]
        coords = np.zeros(n2, dtype=center_coords.dtype)
        coords[0:-1] = center_coords
        coords[-1] = coords[-2] + deltas[-1]
        coords -= deltas // 2
    else:
        raise ValueError(f"cannot determine cell boundaries for"
                         f" coordinate variable {coord_var_name!r} of size {coord_var.size}")

    if np.issubdtype(coords.dtype, np.datetime64):
        coords = coord_bounds_var.astype(np.uint64)
    if np.issubdtype(coord_value.dtype, np.datetime64):
        coord_value = coord_value.astype(np.uint64)
    x = np.linspace(0.0, n1, n2, dtype=np.float64)
    index = np.interp(coord_value, coords, x, left=-1, right=-1).astype(dtype)
    index[index >= n1] = n1 - 1
    return index


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
    report = validate_cube(dataset)
    if report:
        message = f"Dataset" + (name + " " if name else " ")
        message += "is not a valid data cube, because:\n"
        message += "- " + ";\n- ".join(report) + "."
        raise ValueError(message)


def validate_cube(dataset: xr.Dataset) -> List[str]:
    """
    Validate the given *dataset* for being a valid data cube.

    Returns a list of issues, which is empty if *dataset* is a valid data cube.

    :param dataset: A dataset to be validated.
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

    if not np.all(np.diff(var.astype(np.float64)) > 0):
        report.append(f"values of coordinate variable {name!r} must be monotonically increasing")


def _check_time(dataset, name, report):
    var = _check_coord_var(dataset, name, report)
    if var is None:
        return

    if not np.issubdtype(var.dtype, np.datetime64):
        report.append(f"type of coordinate variable {name!r} must be datetime64")

    if not np.all(np.diff(var.astype(np.float64)) > 0):
        report.append(f"values of coordinate variable {name!r} must be monotonically increasing")
