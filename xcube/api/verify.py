# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List

import numpy as np
import xarray as xr


def assert_cube(dataset: xr.Dataset, name=None) -> xr.Dataset:
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

    return dataset


def verify_cube(dataset: xr.Dataset) -> List[str]:
    """
    Verify the given *dataset* for being a valid data cube.

    The tool verifies that *dataset*
    * defines the dimensions "time", "lat", "lon";
    * has corresponding "time", "lat", "lon" coordinate variables and that they
      are valid, e.g. 1-D, non-empty, using correct units;
    * has valid  bounds variables for "time", "lat", "lon" coordinate
      variables, if any;
    * has any data variables and that they are valid, e.g. min. 3-D, all have
      same dimensions, have at least dimensions "time", "lat", "lon".

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
    # TODO (forman): verify bounds coordinate variables
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

    bnds_name = var.attrs.get('bounds', f'{name}_bnds')
    if bnds_name in dataset.coords:
        bnds_var = dataset.coords[bnds_name]
        expected_shape = var.size, 2
        expected_dtype = var.dtype
        if len(bnds_var.dims) != 2 or bnds_var.dims[0] != name:
            report.append(f"bounds coordinate variable {bnds_name!r}"
                          f" must have dimensions ({name!r}, <bounds_dim>)")
        if bnds_var.shape != expected_shape:
            report.append(
                f"shape of bounds coordinate variable {bnds_name!r}"
                f" must be {expected_shape!r} but was {bnds_var.shape!r}")
        if bnds_var.dtype != expected_dtype:
            report.append(
                f"type of bounds coordinate variable {bnds_name!r}"
                f" must be {expected_dtype!r} but was {bnds_var.dtype!r}")
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

    # TODO (forman): the following check is not valid for "lat" because we currently use 'wrong' lat-order
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
