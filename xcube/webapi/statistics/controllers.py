# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Union

import numpy as np
import pyproj
import shapely
import shapely.ops
import xarray as xr

from xcube.constants import CRS_CRS84, LOG
from xcube.core.geom import (
    get_dataset_geometry,
    mask_dataset_by_geometry,
    normalize_geometry,
)
from xcube.core.varexpr import VarExprContext, split_var_assignment
from xcube.server.api import ApiError
from xcube.util.perf import measure_time_cm

from .context import StatisticsContext

NAN_RESULT = {"count": 0}
NAN_RESULT_COMPACT = {}
DEFAULT_BIN_COUNT = 100


def compute_statistics(
    ctx: StatisticsContext,
    ds_id: str,
    var_name: str,
    geometry: Union[dict[str, Any], tuple[float, float]],
    time_label: str,
    trace_perf: bool = False,
):
    measure_time = measure_time_cm(logger=LOG, disabled=not trace_perf)
    with measure_time("Computing statistics"):
        return _compute_statistics(
            ctx, ds_id, var_name, time_label, geometry, DEFAULT_BIN_COUNT
        )


def _compute_statistics(
    ctx: StatisticsContext,
    ds_id: str,
    var_name_or_assign: str,
    time_label: str,
    geometry: Union[dict[str, Any], tuple[float, float]],
    bin_count: int,
):
    ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_id)
    dataset = ml_dataset.get_dataset(0)
    grid_mapping = ml_dataset.grid_mapping

    dataset_contains_time = "time" in dataset

    if dataset_contains_time:
        if time_label is not None:
            try:
                time = np.array(time_label, dtype=dataset.time.dtype)
                dataset = dataset.sel(time=time, method="nearest")
            except (TypeError, ValueError) as e:
                raise ApiError.BadRequest("Invalid query parameter 'time'") from e
        else:
            raise ApiError.BadRequest("Missing query parameter 'time'")
    elif time_label is not None:
        raise ApiError.BadRequest(
            "Query parameter 'time' must not be given"
            " since dataset does not contain a 'time' dimension"
        )

    if isinstance(geometry, tuple):
        compact_mode = True
        geometry = shapely.geometry.Point(geometry)
    else:
        compact_mode = False
        try:
            geometry = normalize_geometry(geometry)
        except (TypeError, ValueError, AttributeError) as e:
            raise ApiError.BadRequest(
                f"Invalid GeoJSON geometry encountered: {{e}}"
            ) from e

    if geometry is not None and not grid_mapping.crs.is_geographic:
        project = pyproj.Transformer.from_crs(
            CRS_CRS84, grid_mapping.crs, always_xy=True
        ).transform
        geometry = shapely.ops.transform(project, geometry)

    nan_result = NAN_RESULT_COMPACT if compact_mode else NAN_RESULT

    x_name, y_name = grid_mapping.xy_dim_names
    if isinstance(geometry, shapely.geometry.Point):
        bounds = get_dataset_geometry(dataset)
        if not bounds.contains(geometry):
            return nan_result
        indexers = {x_name: geometry.x, y_name: geometry.y}
        variable = _get_dataset_variable(var_name_or_assign, dataset)
        value = variable.sel(**indexers, method="Nearest").values
        if np.isnan(value):
            return nan_result
        if compact_mode:
            return {"value": float(value)}
        else:
            return {
                "count": 1,
                "minimum": float(value),
                "maximum": float(value),
                "mean": float(value),
                "deviation": 0.0,
            }

    dataset = mask_dataset_by_geometry(dataset, geometry)
    if dataset is None:
        return nan_result

    variable = _get_dataset_variable(var_name_or_assign, dataset)

    count = int(np.count_nonzero(~np.isnan(variable)))
    if count == 0:
        return nan_result

    # note, casting to float forces intended computation
    minimum = float(variable.min())
    maximum = float(variable.max())
    h_values, h_edges = np.histogram(
        variable, bin_count, range=(minimum, maximum), density=True
    )

    return {
        "count": count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": float(variable.mean()),
        "deviation": float(variable.std()),
        "histogram": {
            "values": [float(v) for v in h_values],
            "edges": [float(v) for v in h_edges],
        },
    }


def _get_dataset_variable(var_name_or_assign: str, dataset: xr.Dataset) -> xr.DataArray:
    var_name, var_expr = split_var_assignment(var_name_or_assign)
    if var_expr:
        variable = VarExprContext(dataset).evaluate(var_expr)
        variable.name = var_name
    else:
        var_name = var_name_or_assign
        variable = dataset[var_name]

    return variable
