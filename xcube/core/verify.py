# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import List

import numpy as np
import xarray as xr

from xcube.core.geom import is_lon_lat_dataset
from xcube.core.schema import get_dataset_time_var_name, get_dataset_xy_var_names


def assert_cube(dataset: xr.Dataset, name=None) -> xr.Dataset:
    """Assert that the given *dataset* is a valid xcube dataset.

    Args:
        dataset: The dataset to be validated.
        name: Optional parameter name.

    Raises:
        ValueError, if dataset is not a valid xcube dataset
    """
    report = verify_cube(dataset)
    if report:
        message = f"Dataset" + (name + " " if name else " ")
        message += "is not a valid xcube dataset, because:\n"
        message += "- " + ";\n- ".join(report) + "."
        raise ValueError(message)

    return dataset


def verify_cube(dataset: xr.Dataset) -> list[str]:
    """Verify the given *dataset* for being a valid xcube dataset.

    The tool verifies that *dataset*
    * defines two spatial x,y coordinate variables, that are 1D, non-empty, using correct units;
    * defines a time coordinate variables, that are 1D, non-empty, using correct units;
    * has valid bounds variables for spatial x,y and time coordinate variables, if any;
    * has any data variables and that they are valid, e.g. min. 3-D, all have
      same dimensions, have at least the dimensions dim(time), dim(y), dim(x) in that order.

    Returns a list of issues, which is empty if *dataset* is a valid xcube dataset.

    Args:
        dataset: A dataset to be verified.

    Returns:
        List of issues or empty list.
    """
    report = []

    xy_var_names = get_dataset_xy_var_names(dataset, must_exist=False)
    if xy_var_names is None:
        report.append(f"missing spatial x,y coordinate variables")

    time_var_name = get_dataset_time_var_name(dataset, must_exist=False)
    if time_var_name is None:
        report.append(f"missing time coordinate variable")

    if time_var_name:
        _check_time(dataset, time_var_name, report)
    if xy_var_names and is_lon_lat_dataset(dataset, xy_var_names=xy_var_names):
        _check_lon_or_lat(dataset, xy_var_names[0], -180.0, 180.0, report)
        _check_lon_or_lat(dataset, xy_var_names[1], -90.0, 90.0, report)

    if xy_var_names and time_var_name:
        _check_data_variables(dataset, xy_var_names, time_var_name, report)

    if xy_var_names:
        _check_coord_equidistance(dataset, xy_var_names[0], xy_var_names[0], report)
        _check_coord_equidistance(dataset, xy_var_names[1], xy_var_names[1], report)

    return report


def _check_coord_equidistance(dataset, coord_name, dim_name, report, rtol=None):
    diff = dataset[coord_name].diff(dim=dim_name)
    if not _check_equidistance_from_diff(dataset, diff, rtol=rtol):
        report.append(f"coordinate variable {coord_name!r} is not equidistant")

    bnds_name = dataset.attrs.get("bounds", f"{coord_name}_bnds")

    if bnds_name in dataset.coords:
        diff = dataset[bnds_name].diff(dim=dim_name)
        if not _check_equidistance_from_diff(dataset, diff[:, 0], rtol=rtol):
            report.append(f"coordinate variable {bnds_name!r} is not equidistant")
        elif not _check_equidistance_from_diff(dataset, diff[:, 1], rtol=rtol):
            report.append(f"coordinate variable {bnds_name!r} is not equidistant")


def _check_equidistance_from_diff(dataset, diff, rtol=None):
    if rtol is None:
        rtol = np.abs(np.divide(diff[0], 100.00)).values

    # Check whether the bounding box intersect with the anti-meridian for lon/lat datasets.
    # This is the case when exactly one difference is negative.
    if is_lon_lat_dataset(dataset):
        ct_neg = diff.where(diff < 0).count().values
        if ct_neg == 1:
            # If a bounding box intersects with the anti-meridian, compute its correct width
            diff = xr.where(diff < 0, diff + 360.0, diff)
    return np.allclose(diff[0], diff, rtol=rtol)


def _check_data_variables(dataset, xy_var_names, time_var_name, report):
    x_var_name, y_var_name = xy_var_names
    x_var, y_var, time_var = (
        dataset[x_var_name],
        dataset[y_var_name],
        dataset[time_var_name],
    )
    x_dim, y_dim, time_dim = x_var.dims[0], y_var.dims[0], time_var.dims[0]

    first_var = None
    first_dims = None
    first_chunks = None
    for var_name, var in dataset.data_vars.items():
        dims = var.dims
        chunks = var.data.chunks if hasattr(var.data, "chunks") else None

        if (
            "grid_mapping_name" in var.attrs
            or "geographic_crs_name" in var.attrs
            or "crs_wkt" in var.attrs
        ):
            # potential CRS / grid mapping variable
            continue

        if (
            len(dims) < 3
            or dims[0] != time_dim
            or dims[-2] != y_dim
            or dims[-1] != x_dim
        ):
            report.append(
                f"dimensions of data variable {var_name!r}"
                f" must be ({time_dim!r}, ..., {y_dim!r}, {x_dim!r}),"
                f" but were {dims!r} for {var_name!r}"
            )

        if first_var is None:
            first_var = var
            first_dims = dims
            first_chunks = chunks
            continue

        if first_dims != dims:
            report.append(
                "dimensions of all data variables must be same,"
                f" but found {first_dims!r} for {first_var.name!r} "
                f"and {dims!r} for {var_name!r}"
            )

        if first_chunks != chunks:
            report.append(
                "all data variables must have same chunk sizes,"
                f" but found {first_chunks!r} for {first_var.name!r} "
                f"and {chunks!r} for {var_name!r}"
            )


def _check_dim(dataset, name, report):
    if name not in dataset.sizes:
        report.append(f"missing dimension {name!r}")
    elif dataset.sizes[name] < 0:
        report.append(f"size of dimension {name!r} must be a positive integer")


def _check_coord_var(dataset, var_name, report):
    if var_name not in dataset.coords:
        report.append(f"missing coordinate variable {var_name!r}")
        return None

    var = dataset.coords[var_name]
    if var.dims != (var_name,):
        report.append(
            f"coordinate variable {var_name!r} must have a single dimension {var_name!r}"
        )
        return None

    if var.size == 0:
        report.append(f"coordinate variable {var_name!r} must not be empty")
        return None

    bnds_name = var.attrs.get("bounds", f"{var_name}_bnds")
    if bnds_name in dataset.coords:
        bnds_var = dataset.coords[bnds_name]
        expected_shape = var.size, 2
        expected_dtype = var.dtype
        if len(bnds_var.dims) != 2 or bnds_var.dims[0] != var_name:
            report.append(
                f"bounds coordinate variable {bnds_name!r}"
                f" must have dimensions ({var_name!r}, <bounds_dim>)"
            )
        if bnds_var.shape != expected_shape:
            report.append(
                f"shape of bounds coordinate variable {bnds_name!r}"
                f" must be {expected_shape!r} but was {bnds_var.shape!r}"
            )
        if bnds_var.dtype != expected_dtype:
            report.append(
                f"type of bounds coordinate variable {bnds_name!r}"
                f" must be {expected_dtype!r} but was {bnds_var.dtype!r}"
            )
        return None

    return var


def _check_lon_or_lat(dataset, var_name, min_value, max_value, report):
    var = _check_coord_var(dataset, var_name, report)
    if var is None:
        return

    if not np.all(np.isfinite(var)):
        report.append(f"values of coordinate variable {var_name!r} must be finite")

    if np.min(var) < min_value or np.max(var) > max_value:
        report.append(
            f"values of coordinate variable {var_name!r}"
            f" must be in the range {min_value} to {max_value}"
        )


def _check_time(dataset, name, report):
    var = _check_coord_var(dataset, name, report)
    if var is None:
        return

    if not np.issubdtype(var.dtype, np.datetime64):
        report.append(f"type of coordinate variable {name!r} must be datetime64")

    if not np.all(np.diff(var.astype(np.float64)) > 0):
        report.append(
            f"values of coordinate variable {name!r} must be monotonic increasing"
        )
