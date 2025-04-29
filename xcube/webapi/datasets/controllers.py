# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import fnmatch
import functools
import io
import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

import matplotlib.colorbar
import matplotlib.colors
import matplotlib.figure
import numpy as np
import pyproj
import xarray as xr

from xcube.constants import CRS_CRS84, LOG
from xcube.core.geom import get_dataset_bounds, get_dataset_geometry
from xcube.core.normalize import DatasetIsNotACubeError
from xcube.core.store import DataStoreError
from xcube.core.tilingscheme import TilingScheme
from xcube.core.timecoord import timestamp_to_iso_string
from xcube.server.api import ApiError

from ..places.controllers import GeoJsonFeatureCollection, find_places
from .authutil import (
    READ_ALL_DATASETS_SCOPE,
    READ_ALL_VARIABLES_SCOPE,
    assert_scopes,
    check_scopes,
)
from .context import DatasetConfig, DatasetsContext

DS_TITLE_ATTR_NAMES = ("title", "name")
DS_DESCRIPTION_ATTR_NAMES = ("description", "abstract", "comment")

VAR_TITLE_ATTR_NAMES = ("title", "name", "long_name")
VAR_DESCRIPTION_ATTR_NAMES = ("description", "abstract", "comment")


def find_dataset_places(
    ctx: DatasetsContext,
    place_group_id: str,
    ds_id: str,
    base_url: str,
    query_expr: Any = None,
    comb_op: str = "and",
) -> GeoJsonFeatureCollection:
    dataset = ctx.get_dataset(ds_id)
    query_geometry = get_dataset_geometry(dataset)
    return find_places(
        ctx.places_ctx,
        place_group_id,
        base_url,
        query_geometry=query_geometry,
        query_expr=query_expr,
        comb_op=comb_op,
    )


def get_datasets(
    ctx: DatasetsContext,
    details: bool = False,
    point: Optional[tuple[float, float]] = None,
    base_url: Optional[str] = None,
    granted_scopes: Optional[set[str]] = None,
) -> dict:
    can_authenticate = ctx.can_authenticate
    # If True, we can shorten scope checking
    if granted_scopes is None:
        can_read_all_datasets = False
    else:
        can_read_all_datasets = READ_ALL_DATASETS_SCOPE in granted_scopes

    LOG.info(f"Collecting datasets for granted scopes {granted_scopes!r}")

    dataset_configs = list(ctx.get_dataset_configs())
    entrypoint_dataset_id = ctx.get_entrypoint_dataset_id()

    dataset_groups = ctx.get_dataset_groups_configs()

    group_info_map = {
        group["Identifier"]: {
            **group,
            "GroupOrder": idx,
        }
        for idx, group in enumerate(dataset_groups)
    }

    dataset_dicts = list()
    for dataset_config in dataset_configs:
        ds_id = dataset_config["Identifier"]

        if dataset_config.get("Hidden"):
            continue

        if (
            can_authenticate
            and not can_read_all_datasets
            and not _allow_dataset(ctx, dataset_config, granted_scopes, check_scopes)
        ):
            LOG.info(f"Rejected dataset {ds_id!r} due to missing permission")
            continue

        ds = ctx.get_dataset(ds_id)
        dataset_dict = dict(id=ds_id)
        _update_dataset_desc_properties(ds, dataset_config, dataset_dict)

        ds_bbox = dataset_config.get("BoundingBox")
        if ds_bbox is not None:
            # Note, dataset_config is validated
            dataset_dict["bbox"] = ds_bbox

        sort_value = dataset_config.get("SortValue")
        if sort_value:
            dataset_dict["sortValue"] = sort_value

        group_id = dataset_config.get("GroupId")
        if group_id:
            if dataset_config.get("GroupTitle"):
                raise ValueError(
                    "GroupTitle should be provided in Dataset "
                    "configuration when using DatasetGroups. Please use the Title "
                    "property in DatasetGroups instead."
                )
            dataset_dict["groupId"] = group_id
            if group_id in group_info_map:
                group_info = group_info_map[group_id]
                dataset_dict["groupOrder"] = group_info["GroupOrder"]
                dataset_dict["groupTitle"] = group_info["Title"]
                if group_info.get("Description"):
                    dataset_dict["groupDescription"] = group_info["Description"]

        LOG.info(f"Collected dataset {ds_id}")
        dataset_dicts.append(dataset_dict)

    # Important note:
    # the "point" parameter is used by
    # the CyanoAlert app only

    if details or point:
        filtered_dataset_dicts = []
        for dataset_dict in dataset_dicts:
            ds_id = dataset_dict["id"]
            try:
                if point:
                    if "bbox" not in dataset_dict:
                        dataset_dict["bbox"] = list(get_dataset_bounds(ds))
                if details:
                    LOG.info(f"Loading details for dataset {ds_id}")
                    dataset_dict.update(
                        get_dataset(ctx, ds_id, base_url, granted_scopes=granted_scopes)
                    )
                filtered_dataset_dicts.append(dataset_dict)
            except Exception as e:
                LOG.warning(f"Skipping dataset {ds_id}: {e}", exc_info=True)
        dataset_dicts = filtered_dataset_dicts

    if point:
        is_point_in_dataset_bbox = functools.partial(_is_point_in_dataset_bbox, point)
        # noinspection PyTypeChecker
        dataset_dicts = list(filter(is_point_in_dataset_bbox, dataset_dicts))

    if not dataset_dicts:
        LOG.warning("No datasets provided for current user.")

    return dict(datasets=dataset_dicts, entrypointDatasetId=entrypoint_dataset_id)


def get_dataset(
    ctx: DatasetsContext,
    ds_id: str,
    base_url: Optional[str] = None,
    granted_scopes: Optional[set[str]] = None,
) -> dict:
    can_authenticate = ctx.can_authenticate
    # If True, we can shorten scope checking
    if granted_scopes is None:
        can_read_all_datasets = False
        can_read_all_variables = False
    else:
        can_read_all_datasets = READ_ALL_DATASETS_SCOPE in granted_scopes
        can_read_all_variables = READ_ALL_VARIABLES_SCOPE in granted_scopes

    dataset_config = ctx.get_dataset_config(ds_id)
    ds_id = dataset_config["Identifier"]

    if can_authenticate and not can_read_all_datasets:
        _allow_dataset(ctx, dataset_config, granted_scopes, assert_scopes)

    try:
        ml_ds = ctx.get_ml_dataset(ds_id)
    except (ValueError, DataStoreError) as e:
        raise DatasetIsNotACubeError(f"could not open dataset: {e}") from e

    ds = ml_ds.get_dataset(0)
    dataset_dict = dict(id=ds_id)
    _update_dataset_desc_properties(ds, dataset_config, dataset_dict)

    try:
        ts_ds = ctx.get_time_series_dataset(ds_id)
    except (ValueError, DataStoreError):
        ts_ds = None

    x_name, y_name = ml_ds.grid_mapping.xy_dim_names

    crs = ml_ds.grid_mapping.crs
    transformer = pyproj.Transformer.from_crs(crs, CRS_CRS84, always_xy=True)
    dataset_bounds = get_dataset_bounds(ds)

    ds_bbox = dataset_config.get("BoundingBox")
    if ds_bbox is not None:
        # Note, JSON-Schema already verified that ds_bbox is valid.
        # 'BoundingBox' is always given in geographical coordinates.
        ds_bbox = list(ds_bbox)
    else:
        x1, y1, x2, y2 = dataset_bounds
        if not crs.is_geographic:
            (x1, x2), (y1, y2) = transformer.transform((x1, x2), (y1, y2))
        ds_bbox = [x1, y1, x2, y2]

    dataset_dict["bbox"] = list(ds_bbox)
    dataset_dict["geometry"] = get_bbox_geometry(dataset_bounds, transformer)
    dataset_dict["spatialRef"] = crs.to_string()

    variable_dicts = []
    dim_names = set()

    tiling_scheme = ml_ds.derive_tiling_scheme(TilingScheme.GEOGRAPHIC)
    LOG.debug(
        "Tile level range for dataset %s: %d to %d",
        ds_id,
        tiling_scheme.min_level,
        tiling_scheme.max_level,
    )

    spatial_var_names = [
        str(var_name)
        for var_name, var in ds.data_vars.items()
        if (len(var.dims) >= 2 and var.dims[-2] == y_name and var.dims[-1] == x_name)
    ]

    var_name_patterns = dataset_config.get("Variables")
    if var_name_patterns:
        spatial_var_names = filter_variable_names(spatial_var_names, var_name_patterns)
        if not spatial_var_names:
            LOG.warning(
                "No variable matched any of the patterns given"
                ' in the "Variables" filter.'
                ' You may specify a wildcard "*" as last item.'
            )

    for var_name in spatial_var_names:
        var = ds.data_vars[var_name]

        if (
            can_authenticate
            and not can_read_all_variables
            and not _allow_variable(ctx, dataset_config, var_name, granted_scopes)
        ):
            continue

        var_title, var_description = get_variable_title_and_description(var_name, var)
        variable_dict = dict(
            id=f"{ds_id}.{var_name}",
            name=var_name,
            dims=list(var.dims),
            shape=list(var.shape),
            dtype=str(var.dtype),
            units=var.attrs.get("units", ""),
            title=var_title,
            timeChunkSize=get_time_chunk_size(ts_ds, var_name, ds_id),
        )
        if var_description:
            variable_dict["description"] = var_description

        tile_url = _get_dataset_tile_url2(ctx, ds_id, var_name, base_url)
        # Note that tileUrl is no longer used since xcube viewer v0.13
        variable_dict["tileUrl"] = tile_url
        LOG.debug("Tile URL for variable %s: %s", var_name, tile_url)

        variable_dict["tileLevelMin"] = tiling_scheme.min_level
        variable_dict["tileLevelMax"] = tiling_scheme.max_level

        cmap_name, cmap_norm, (cmap_vmin, cmap_vmax) = ctx.get_color_mapping(
            ds_id, var_name
        )
        variable_dict["colorBarName"] = cmap_name
        variable_dict["colorBarNorm"] = cmap_norm
        variable_dict["colorBarMin"] = cmap_vmin
        variable_dict["colorBarMax"] = cmap_vmax

        if hasattr(var.data, "_repr_html_"):
            # noinspection PyProtectedMember
            variable_dict["htmlRepr"] = var.data._repr_html_()

        variable_dict["attrs"] = {
            key: ("NaN" if isinstance(value, float) and np.isnan(value) else value)
            for key, value in var.attrs.items()
        }

        variable_dicts.append(variable_dict)
        for dim_name in var.dims:
            dim_names.add(dim_name)

    dataset_dict["variables"] = variable_dicts

    if not variable_dicts:
        if not ds.data_vars:
            message = f"Dataset {ds_id!r} has no variables"
        else:
            message = f"Dataset {ds_id!r} has no published variables"
        raise DatasetIsNotACubeError(message)

    rgb_var_names, rgb_norm_ranges = ctx.get_rgb_color_mapping(ds_id)
    if any(rgb_var_names):
        rgb_tile_url = _get_dataset_tile_url2(ctx, ds_id, "rgb", base_url)
        rgb_schema = {
            "varNames": rgb_var_names,
            "normRanges": rgb_norm_ranges,
            # Note that tileUrl is no longer used since xcube viewer v0.13
            "tileUrl": rgb_tile_url,
            "tileLevelMin": tiling_scheme.min_level,
            "tileLevelMax": tiling_scheme.max_level,
        }

        dataset_dict["rgbSchema"] = rgb_schema

    dataset_dict["dimensions"] = [
        get_dataset_coordinates(ctx, ds_id, str(dim_name)) for dim_name in dim_names
    ]

    dataset_dict["attrs"] = {
        key: ds.attrs[key] for key in sorted(list(map(str, ds.attrs.keys())))
    }

    dataset_attributions = dataset_config.get(
        "Attribution", ctx.config.get("DatasetAttribution")
    )
    if dataset_attributions is not None:
        if isinstance(dataset_attributions, str):
            dataset_attributions = [dataset_attributions]
        dataset_dict["attributions"] = dataset_attributions

    place_groups = ctx.get_dataset_place_groups(ds_id, base_url)
    if place_groups:
        dataset_dict["placeGroups"] = _filter_place_groups(
            place_groups, del_features=True
        )

    return dataset_dict


def get_dataset_title_and_description(
    dataset: xr.Dataset,
    dataset_config: Mapping[str, Any] | None = None,
) -> tuple[str, str | None]:
    dataset_config = dataset_config or {}
    ds_title = dataset_config.get(
        "Title",
        _get_str_attr(
            dataset.attrs,
            DS_TITLE_ATTR_NAMES,
            dataset_config.get("Identifier"),
        ),
    )
    ds_description = dataset_config.get(
        "Description",
        _get_str_attr(dataset.attrs, DS_DESCRIPTION_ATTR_NAMES),
    )
    return ds_title or "", ds_description or None


def get_variable_title_and_description(
    var_name: str,
    var: xr.DataArray,
) -> tuple[str, str | None]:
    var_title = _get_str_attr(var.attrs, VAR_TITLE_ATTR_NAMES, var_name)
    var_description = _get_str_attr(var.attrs, VAR_DESCRIPTION_ATTR_NAMES)
    return var_title or "", var_description or None


def _update_dataset_desc_properties(
    ds: xr.Dataset, dataset_config: Mapping[str, Any], dataset_dict: dict[str, Any]
):
    ds_title, ds_description = get_dataset_title_and_description(ds, dataset_config)
    group_title = dataset_config.get("GroupTitle")
    tags = dataset_config.get("Tags")

    dataset_dict["title"] = ds_title
    if ds_description:
        dataset_dict["description"] = ds_description
    if group_title:
        dataset_dict["groupTitle"] = group_title
    if tags:
        dataset_dict["tags"] = tags


def filter_variable_names(
    var_names: list[str], var_name_patterns: list[str]
) -> list[str]:
    filtered_var_names = []
    filtered_var_names_set = set()
    for var_name_pattern in var_name_patterns:
        for var_name in var_names:
            if var_name in filtered_var_names_set:
                continue
            if var_name == var_name_pattern or fnmatch.fnmatch(
                var_name, var_name_pattern
            ):
                filtered_var_names.append(var_name)
                filtered_var_names_set.add(var_name)
    return filtered_var_names


def get_bbox_geometry(
    dataset_bounds: tuple[float, float, float, float],
    transformer: pyproj.Transformer,
    n: int = 6,
):
    x1, y1, x2, y2 = dataset_bounds
    x_coords = []
    y_coords = []
    x_coords.extend(np.full(n - 1, x1))
    y_coords.extend(np.linspace(y1, y2, n)[: n - 1])
    x_coords.extend(np.linspace(x1, x2, n)[: n - 1])
    y_coords.extend(np.full(n - 1, y2))
    x_coords.extend(np.full(n - 1, x2))
    y_coords.extend(np.linspace(y2, y1, n)[: n - 1])
    x_coords.extend(np.linspace(x2, x1, n)[: n - 1])
    y_coords.extend(np.full(n - 1, y1))
    x_coords, y_coords = transformer.transform(x_coords, y_coords)
    coordinates = list(map(list, zip(map(float, x_coords), map(float, y_coords))))
    coordinates.append(coordinates[0])
    geometry = dict(type="Polygon", coordinates=[coordinates])
    return geometry


def get_time_chunk_size(
    ts_ds: Optional[xr.Dataset], var_name: str, ds_id: str
) -> Optional[int]:
    """Get the time chunk size for variable *var_name*
    in time-chunked dataset *ts_ds*.

    Internal function.

    Args:
        ts_ds: time-chunked dataset
        var_name: variable name
        ds_id: original dataset identifier

    Returns:
        the time chunk size (integer) or None
    """
    if ts_ds is not None:
        ts_var: Optional[xr.DataArray] = ts_ds.get(var_name)
        if ts_var is not None:
            chunks = ts_var.chunks
            if chunks is None:
                LOG.warning(
                    f"variable {var_name!r}"
                    f" in time-chunked dataset {ds_id!r}"
                    f" is not chunked"
                )
                return None
            try:
                time_index = ts_var.dims.index("time")
                time_chunks = chunks[time_index]
            except ValueError:
                time_chunks = None
            if not time_chunks:
                LOG.warning(
                    f"no chunks found"
                    f" for dimension 'time'"
                    f" of variable {var_name!r}"
                    f" in time-chunked dataset {ds_id!r}"
                )
                return None
            if len(time_chunks) == 1:
                return time_chunks[0]
            return max(*time_chunks)
        else:
            LOG.warning(
                f"variable {var_name!r} not found in time-chunked dataset {ds_id!r}"
            )
    return None


def _allow_dataset(
    ctx: DatasetsContext,
    dataset_config: DatasetConfig,
    granted_scopes: Optional[set[str]],
    function: Callable[[set, Optional[set], bool], Any],
) -> Any:
    required_scopes = ctx.get_required_dataset_scopes(dataset_config)
    # noinspection PyArgumentList
    return function(
        required_scopes, granted_scopes, is_substitute=_is_substitute(dataset_config)
    )


def _allow_variable(
    ctx: DatasetsContext,
    dataset_config: DatasetConfig,
    var_name: str,
    granted_scopes: Optional[set[str]],
) -> bool:
    required_scopes = ctx.get_required_variable_scopes(dataset_config, var_name)
    # noinspection PyArgumentList
    return check_scopes(
        required_scopes, granted_scopes, is_substitute=_is_substitute(dataset_config)
    )


def _is_substitute(dataset_config: DatasetConfig) -> bool:
    return dataset_config.get("AccessControl", {}).get("IsSubstitute", False)


def get_dataset_place_groups(
    ctx: DatasetsContext, ds_id: str, base_url: str
) -> list[GeoJsonFeatureCollection]:
    # Do not load or return features, just place group (metadata).
    place_groups = ctx.get_dataset_place_groups(ds_id, base_url, load_features=False)
    return _filter_place_groups(place_groups, del_features=True)


def get_dataset_place_group(
    ctx: DatasetsContext, ds_id: str, place_group_id: str, base_url: str
) -> GeoJsonFeatureCollection:
    # Load and return features for specific place group.
    place_group = ctx.get_dataset_place_group(
        ds_id, place_group_id, base_url, load_features=True
    )
    return _filter_place_group(place_group, del_features=False)


def get_dataset_coordinates(ctx: DatasetsContext, ds_id: str, dim_name: str) -> dict:
    ds, var = ctx.get_dataset_and_coord_variable(ds_id, dim_name)
    if np.issubdtype(var.dtype, np.floating):
        values = list(map(float, var.values))
    elif np.issubdtype(var.dtype, np.integer):
        values = list(map(int, var.values))
    elif len(var) == 1:
        values = [timestamp_to_iso_string(var.values[0], round_fn="floor")]
    else:
        # see https://github.com/dcs4cop/xcube-viewer/issues/289
        assert len(var) > 1, "Dimension length must be greater than 0."
        values = (
            [timestamp_to_iso_string(var.values[0], round_fn="floor")]
            + list(map(timestamp_to_iso_string, var.values[1:-1]))
            + [timestamp_to_iso_string(var.values[-1], round_fn="ceil")]
        )
    return dict(
        name=dim_name, size=len(values), dtype=str(var.dtype), coordinates=values
    )


# noinspection PyUnusedLocal
def get_color_bars(ctx: DatasetsContext, mime_type: str) -> str:
    cmaps = ctx.colormap_registry.to_json()
    if mime_type == "application/json":
        return json.dumps(cmaps, indent=2)
    elif mime_type == "text/html":
        html_head = (
            "<!DOCTYPE html>\n"
            + '<html lang="en">\n'
            + "<head>"
            + '<meta charset="UTF-8">'
            + "<title>xcube Server Colormaps</title>"
            + "</head>\n"
            + '<body style="padding: 0.2em; font-family: sans-serif">\n'
        )
        html_body = ""
        html_foot = "</body>\n</html>\n"
        for cmap_cat, cmap_desc, cmap_bars in cmaps:
            html_body += "    <h2>%s</h2>\n" % cmap_cat
            html_body += "    <p>%s</p>\n" % cmap_desc
            html_body += '    <table style=border: 0">\n'
            for cmap_bar in cmap_bars:
                if len(cmap_bar) == 3:
                    cmap_name, cmap_data, cm_code = cmap_bar
                else:
                    cmap_name, cmap_data = cmap_bar
                cmap_image = (
                    f'<img src="data:image/png;base64,{cmap_data}"'
                    f' width="100%%"'
                    f' height="24"/>'
                )
                name_cell = f'<td style="width: 5em"><code>{cmap_name}</code></td>'
                image_cell = f'<td style="width: 40em">{cmap_image}</td>'
                row = f"<tr>{name_cell}{image_cell}</tr>\n"
                html_body += f"        {row}\n"
            html_body += "    </table>\n"
        return html_head + html_body + html_foot
    raise ApiError.BadRequest(f"Format {mime_type!r} not supported for colormaps")


def _is_point_in_dataset_bbox(point: tuple[float, float], dataset_dict: dict):
    if "bbox" not in dataset_dict:
        return False
    x, y = point
    x_min, y_min, x_max, y_max = dataset_dict["bbox"]
    if not (y_min <= y <= y_max):
        return False
    if x_min < x_max:
        return x_min <= x <= x_max
    else:
        # Bounding box crosses anti-meridian
        return x_min <= x <= 180.0 or -180.0 <= x <= x_max


def _filter_place_group(place_group: dict, del_features: bool = False) -> dict:
    place_group = dict(place_group)
    del place_group["sourcePaths"]
    del place_group["sourceEncoding"]
    if del_features:
        del place_group["features"]
    return place_group


def _filter_place_groups(place_groups, del_features: bool = False) -> list[dict]:
    if del_features:

        def __filter_place_group(place_group):
            return _filter_place_group(place_group, del_features=True)

    else:

        def __filter_place_group(place_group):
            return _filter_place_group(place_group, del_features=False)

    return list(map(__filter_place_group, place_groups))


def _get_dataset_tile_url2(
    ctx: DatasetsContext, ds_id: str, var_name: str, base_url: str
):
    import urllib.parse

    return ctx.get_service_url(
        base_url,
        "datasets",
        urllib.parse.quote_plus(ds_id),
        "vars",
        urllib.parse.quote_plus(var_name),
        "tiles2",
        "{z}/{y}/{x}",
    )


def get_legend(
    ctx: DatasetsContext, ds_id: str, var_name: str, params: Mapping[str, str]
):
    default_cmap_cbar, default_cmap_norm, (default_cmap_vmin, default_cmap_vmax) = (
        ctx.get_color_mapping(ds_id, var_name)
    )

    try:
        cmap_name = params.get("cmap", params.get("cbar", default_cmap_cbar))
        cmap_norm = params.get("norm", str(default_cmap_norm))
        cmap_vmin = float(params.get("vmin", str(default_cmap_vmin)))
        cmap_vmax = float(params.get("vmax", str(default_cmap_vmax)))
        cmap_w = int(params.get("width", "256"))
        cmap_h = int(params.get("height", "16"))
    except (ValueError, TypeError):
        raise ApiError.BadRequest("Invalid color legend parameter(s)")

    cmap, colormap = ctx.colormap_registry.get_cmap(cmap_name)
    assert colormap is not None

    norm = None
    if cmap_norm == "lin":
        norm = matplotlib.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
    elif cmap_norm == "log":
        norm = matplotlib.colors.LogNorm(vmin=cmap_vmin, vmax=cmap_vmax)
    elif cmap_norm == "cat":
        norm = matplotlib.colors.BoundaryNorm(list(range(cmap.N + 1)), ncolors=cmap.N)
    if norm is None:
        norm = colormap.norm
    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)

    fig = matplotlib.figure.Figure(figsize=(cmap_w, cmap_h))
    ax1 = fig.add_subplot(1, 1, 1)
    image_legend = matplotlib.colorbar.ColorbarBase(
        ax1, format="%.1f", ticks=None, cmap=cmap, norm=norm, orientation="vertical"
    )

    image_legend_label = ctx.get_legend_label(ds_id, var_name)
    if image_legend_label is not None:
        image_legend.set_label(image_legend_label)

    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")

    return buffer.getvalue()


def _get_str_attr(
    attrs: Mapping[str, Any], keys: Sequence[str], default: Optional[str] = None
) -> Optional[str]:
    for k in keys:
        v = attrs.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return default if isinstance(default, str) else None
