# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import warnings
from typing import Dict, Any, Union
from collections.abc import Sequence

import numpy as np
import xarray as xr

from xcube.core.schema import CubeSchema
from xcube.core.select import select_variables_subset
from xcube.core.verify import assert_cube


def resample_in_time(
    dataset: xr.Dataset,
    frequency: str,
    method: Union[str, Sequence[str]],
    offset=None,
    tolerance=None,
    interp_kind=None,
    time_chunk_size=None,
    var_names: Sequence[str] = None,
    metadata: dict[str, Any] = None,
    cube_asserted: bool = False,
) -> xr.Dataset:
    """Resample a dataset in the time dimension.

    The argument *method* may be one or a sequence of
    ``'all'``, ``'any'``,
    ``'argmax'``, ``'argmin'``, ``'count'``,
    ``'first'``, ``'last'``,
    ``'max'``, ``'min'``, ``'mean'``, ``'median'``,
    ``'percentile_<p>'``,
    ``'std'``, ``'sum'``, ``'var'``.

    In value ``'percentile_<p>'`` is a placeholder,
    where ``'<p>'`` must be replaced by an integer percentage
    value, e.g. ``'percentile_90'`` is the 90%-percentile.

    *Important note:* As of xarray 0.14 and dask 2.8, the
    methods ``'median'`` and ``'percentile_<p>'` cannot be
    used if the variables in *cube* comprise chunked dask arrays.
    In this case, use the ``compute()`` or ``load()`` method
    to convert dask arrays into numpy arrays.

    Args:
        dataset: The xcube dataset.
        frequency: Temporal aggregation frequency. Use format
            "<count><offset>" where <offset> is one of 'H', 'D', 'W',
            'M', 'Q', 'Y'.
        method: Resampling method or sequence of resampling methods.
        offset: Offset used to adjust the resampled time labels. Uses
            same syntax as *frequency*.
        time_chunk_size: If not None, the chunk size to be used for the
            "time" dimension.
        var_names: Variable names to include.
        tolerance: Time tolerance for selective upsampling methods.
            Defaults to *frequency*.
        interp_kind: Kind of interpolation if *method* is
            'interpolation'.
        metadata: Output metadata.
        cube_asserted: If False, *cube* will be verified, otherwise it
            is expected to be a valid cube.

    Returns:
        A new xcube dataset resampled in time.
    """
    if not cube_asserted:
        assert_cube(dataset)

    if frequency == "all":
        time_gap = np.array(dataset.time[-1]) - np.array(dataset.time[0])
        days = int((np.timedelta64(time_gap, "D") / np.timedelta64(1, "D")) + 1)
        frequency = f"{days}D"

    if var_names:
        dataset = select_variables_subset(dataset, var_names)

    resampler = dataset.resample(
        skipna=True, closed="left", label="left", time=frequency, loffset=offset
    )

    if isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)

    percentile_prefix = "percentile_"

    resampled_cubes = []
    for method in methods:
        method_args = []
        method_postfix = method
        if method.startswith(percentile_prefix):
            p = int(method[len(percentile_prefix) :])
            q = p / 100.0
            method_args = [q]
            method_postfix = f"p{p}"
            method = "quantile"
        resampling_method = getattr(resampler, method)
        method_kwargs = get_method_kwargs(method, frequency, interp_kind, tolerance)
        resampled_cube = resampling_method(*method_args, **method_kwargs)
        resampled_cube = resampled_cube.rename(
            {
                var_name: f"{var_name}_{method_postfix}"
                for var_name in resampled_cube.data_vars
            }
        )
        resampled_cubes.append(resampled_cube)

    if len(resampled_cubes) == 1:
        resampled_cube = resampled_cubes[0]
    else:
        resampled_cube = xr.merge(resampled_cubes)

    # TODO: add time_bnds to resampled_ds
    time_coverage_start = "%s" % dataset.time[0]
    time_coverage_end = "%s" % dataset.time[-1]

    resampled_cube.attrs.update(metadata or {})
    # TODO: add other time_coverage_ attributes
    resampled_cube.attrs.update(
        time_coverage_start=time_coverage_start, time_coverage_end=time_coverage_end
    )

    schema = CubeSchema.new(dataset)
    chunk_sizes = {schema.dims[i]: schema.chunks[i] for i in range(schema.ndim)}

    if isinstance(time_chunk_size, int) and time_chunk_size >= 0:
        chunk_sizes["time"] = time_chunk_size

    return resampled_cube.chunk(chunk_sizes)


def get_method_kwargs(method, frequency, interp_kind, tolerance):
    if method == "interpolate":
        kwargs = {"kind": interp_kind or "linear"}
    elif method in {"nearest", "bfill", "ffill", "pad"}:
        kwargs = {"tolerance": tolerance or frequency}
    elif method in {
        "first",
        "last",
        "sum",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "var",
    }:
        kwargs = {"dim": "time", "keep_attrs": True, "skipna": True}
    elif method == "prod":
        kwargs = {"dim": "time", "skipna": True}
    elif method == "count":
        kwargs = {"dim": "time", "keep_attrs": True}
    else:
        kwargs = {}
    return kwargs
