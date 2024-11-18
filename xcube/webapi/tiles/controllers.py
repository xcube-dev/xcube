# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional, Dict
from collections.abc import Mapping

from xcube.constants import LOG
from xcube.core.tile import DEFAULT_CRS_NAME
from xcube.core.tile import DEFAULT_FORMAT
from xcube.core.tile import TileNotFoundException
from xcube.core.tile import TileRequestException
from xcube.core.tile import compute_rgba_tile
from xcube.core.tilingscheme import DEFAULT_TILE_SIZE
from xcube.server.api import ApiError
from xcube.util.perf import measure_time_cm
from .context import TilesContext


def compute_ml_dataset_tile(
    ctx: TilesContext,
    ds_id: str,
    var_name: str,
    crs_name: Optional[str],
    x: str,
    y: str,
    z: str,
    params: Mapping[str, str],
):
    params = dict(params)
    trace_perf = params.pop("debug", "1" if ctx.datasets_ctx.trace_perf else "0") == "1"
    measure_time = measure_time_cm(logger=LOG, disabled=not trace_perf)
    with measure_time("Computing RGBA tile"):
        return _compute_ml_dataset_tile(
            ctx, ds_id, var_name, crs_name, x, y, z, params, trace_perf
        )


def _compute_ml_dataset_tile(
    ctx: TilesContext,
    ds_id: str,
    var_name: str,
    crs_name: Optional[str],
    x: str,
    y: str,
    z: str,
    args: dict[str, str],
    trace_perf: bool,
):
    try:
        x, y, z = (int(c) for c in (x, y, z))
    except ValueError:
        raise ApiError.BadRequest("x, y, z must be integers")

    crs_name = args.pop("crs", crs_name or DEFAULT_CRS_NAME)
    retina = args.pop("retina", None) == "1"
    cmap_name = args.pop("cmap", args.pop("cbar", None))
    cmap_norm = args.pop("norm", None)
    value_min = float(args.pop("vmin", 0.0))
    value_max = float(args.pop("vmax", 1.0))
    format = args.pop("format", DEFAULT_FORMAT)

    if format not in ("png", "image/png"):
        raise ApiError.BadRequest(f"Illegal format {format!r}")

    ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_id)
    if var_name == "rgb":
        var_names, value_ranges = ctx.datasets_ctx.get_rgb_color_mapping(
            ds_id, norm_range=(value_min, value_max)
        )
        components = "r", "g", "b"
        for i, c in enumerate(components):
            var_names[i] = args.pop(c, var_names[i])
            if var_names[i] is None:
                raise ApiError.BadRequest(
                    f"No variable in dataset {ds_id!r}"
                    f" specified for RGB component {c.upper()}"
                )
            value_ranges[i] = (
                float(args.pop(f"{c}vmin", value_ranges[i][0])),
                float(args.pop(f"{c}vmax", value_ranges[i][1])),
            )
    else:
        if (
            cmap_name is None
            or cmap_norm is None
            or value_min is None
            or value_max is None
        ):
            default_cmap_name, default_cmap_norm, default_value_range = (
                ctx.datasets_ctx.get_color_mapping(ds_id, var_name)
            )
            default_value_min, default_value_max = default_value_range
            if cmap_name is None:
                cmap_name = default_cmap_name
            if cmap_norm is None:
                cmap_norm = default_cmap_norm
            if value_min is None:
                value_min = default_value_min
            if value_max is None:
                value_max = default_value_max
        var_names = (var_name,)
        value_ranges = ((value_min, value_max),)

    try:
        return compute_rgba_tile(
            ml_dataset,
            var_names,
            x,
            y,
            z,
            ctx.datasets_ctx.colormap_registry,
            crs_name=crs_name,
            tile_size=(2 if retina else 1) * DEFAULT_TILE_SIZE,
            cmap_name=cmap_name,
            cmap_norm=cmap_norm,
            value_ranges=value_ranges,
            non_spatial_labels=args,
            format=format,
            trace_perf=trace_perf,
        )
    except TileNotFoundException as e:
        raise ApiError.NotFound(f"{e}") from e
    except TileRequestException as e:
        raise ApiError.BadRequest(f"{e}") from e
