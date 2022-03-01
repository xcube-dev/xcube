# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import functools
import json
import math
from typing import Dict, Tuple, List, Set

import numpy as np

from xcube.constants import LOG
from xcube.core.geom import get_dataset_bounds
from xcube.core.normalize import DatasetIsNotACubeError
from xcube.core.store import DataStoreError
from xcube.core.timecoord import timestamp_to_iso_string
from xcube.util.assertions import assert_instance
from xcube.util.cmaps import get_cmaps
from xcube.util.tilegrid import TileGrid
from xcube.webapi.auth import assert_scopes
from xcube.webapi.auth import check_scopes
from xcube.webapi.context import ServiceContext
from xcube.webapi.controllers.places import GeoJsonFeatureCollection
from xcube.webapi.controllers.tiles import get_dataset_tile_url
from xcube.webapi.controllers.tiles import get_tile_source_options
from xcube.webapi.errors import ServiceBadRequestError


def get_datasets(ctx: ServiceContext,
                 details: bool = False,
                 client: str = None,
                 point: Tuple[float, float] = None,
                 base_url: str = None,
                 granted_scopes: Set[str] = None) -> Dict:
    granted_scopes = granted_scopes or set()

    dataset_configs = list(ctx.get_dataset_configs())

    dataset_dicts = list()
    for dataset_config in dataset_configs:

        ds_id = dataset_config['Identifier']

        if dataset_config.get('Hidden'):
            continue

        if 'read:dataset:*' not in granted_scopes:
            required_scopes = ctx.get_required_dataset_scopes(dataset_config)
            is_substitute = dataset_config \
                .get('AccessControl', {}) \
                .get('IsSubstitute', False)
            if not check_scopes(required_scopes,
                                granted_scopes,
                                is_substitute=is_substitute):
                continue

        dataset_dict = dict(id=ds_id)

        dataset_dict['title'] = ds_id
        if 'Title' in dataset_config:
            ds_title = dataset_config['Title']
            if ds_title and isinstance(ds_title, str):
                dataset_dict['title'] = ds_title

        if 'BoundingBox' in dataset_config:
            ds_bbox = dataset_config['BoundingBox']
            if ds_bbox \
                    and len(ds_bbox) == 4 \
                    and all(map(lambda c: isinstance(c, float)
                                          or isinstance(c, int),
                                ds_bbox)):
                dataset_dict['bbox'] = ds_bbox

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
                        ds = ctx.get_dataset(ds_id)
                        if "bbox" not in dataset_dict:
                            dataset_dict["bbox"] = list(get_dataset_bounds(ds))
                    if details:
                        dataset_dict.update(
                            get_dataset(ctx, ds_id, client,
                                        base_url,
                                        granted_scopes=granted_scopes)
                        )
                    filtered_dataset_dicts.append(dataset_dict)
                except (DatasetIsNotACubeError, CubeIsNotDisplayable) as e:
                    LOG.warning(f'skipping dataset {ds_id}: {e}')
            dataset_dicts = filtered_dataset_dicts
    if point:
        is_point_in_dataset_bbox = functools.partial(
            _is_point_in_dataset_bbox, point
        )
        # noinspection PyTypeChecker
        dataset_dicts = list(filter(is_point_in_dataset_bbox, dataset_dicts))

    return dict(datasets=dataset_dicts)


def get_dataset(ctx: ServiceContext,
                ds_id: str,
                client=None,
                base_url: str = None,
                granted_scopes: Set[str] = None) -> Dict:
    granted_scopes = granted_scopes or set()

    dataset_config = ctx.get_dataset_config(ds_id)
    ds_id = dataset_config['Identifier']

    if 'read:dataset:*' not in granted_scopes:
        required_scopes = ctx.get_required_dataset_scopes(dataset_config)
        assert_scopes(required_scopes, granted_scopes or set())

    try:
        ml_ds = ctx.get_ml_dataset(ds_id)
    except (ValueError, DataStoreError) as e:
        raise DatasetIsNotACubeError(f'could not open dataset: {e}') from e

    grid_mapping = ml_ds.grid_mapping
    if not grid_mapping.crs.is_geographic:
        raise CubeIsNotDisplayable(f'CRS is not geographic:'
                                   f' {grid_mapping.crs.srs}')
    if not math.isclose(grid_mapping.x_res,
                        grid_mapping.y_res,
                        rel_tol=0.01):  # we allow up to 1% dev
        raise CubeIsNotDisplayable(f'spatial resolutions are'
                                   f' different in x, y:'
                                   f' {grid_mapping.x_res}'
                                   f' and {grid_mapping.y_res}')
    try:
        # Make sure we have a valid tile grid
        tile_grid = ml_ds.tile_grid
        assert_instance(tile_grid, TileGrid)
    except ValueError as e:
        raise CubeIsNotDisplayable(f'could not create tile grid: {e}')

    ds = ml_ds.get_dataset(0)

    ds_title = dataset_config.get('Title',
                                  ds.attrs.get('title',
                                               ds.attrs.get('name',
                                                            ds_id)))
    dataset_dict = dict(id=ds_id, title=ds_title)

    if "bbox" not in dataset_dict:
        dataset_dict["bbox"] = list(get_dataset_bounds(ds))

    variable_dicts = []
    for var_name in ds.data_vars:
        var = ds.data_vars[var_name]
        dims = var.dims
        if len(dims) < 3 \
                or dims[0] != 'time' \
                or dims[-2] != 'lat' \
                or dims[-1] != 'lon':
            continue

        if 'read:variable:*' not in granted_scopes:
            required_scopes = ctx.get_required_variable_scopes(dataset_config,
                                                               var_name)
            if not check_scopes(required_scopes, granted_scopes):
                continue

        variable_dict = dict(id=f'{ds_id}.{var_name}',
                             name=var_name,
                             dims=list(dims),
                             shape=list(var.shape),
                             dtype=str(var.dtype),
                             units=var.attrs.get('units', ''),
                             title=var.attrs.get('title',
                                                 var.attrs.get('long_name',
                                                               var_name)))

        if client is not None:
            tile_grid = ctx.get_tile_grid(ds_id)
            tile_xyz_source_options = get_tile_source_options(
                tile_grid,
                get_dataset_tile_url(ctx, ds_id,
                                     var_name,
                                     base_url),
                client=client
            )
            variable_dict["tileSourceOptions"] = tile_xyz_source_options

        cmap_name, (cmap_vmin, cmap_vmax) = ctx.get_color_mapping(ds_id,
                                                                  var_name)
        variable_dict["colorBarName"] = cmap_name
        variable_dict["colorBarMin"] = cmap_vmin
        variable_dict["colorBarMax"] = cmap_vmax

        if hasattr(var.data, '_repr_html_'):
            # noinspection PyProtectedMember
            variable_dict["htmlRepr"] = var.data._repr_html_()

        variable_dict["attrs"] = {
            key: ("NaN" if isinstance(value, float)
                           and np.isnan(value) else value)
            for key, value in var.attrs.items()
        }

        variable_dicts.append(variable_dict)

    dataset_dict["variables"] = variable_dicts

    rgb_var_names, rgb_norm_ranges = ctx.get_rgb_color_mapping(ds_id)
    if any(rgb_var_names):
        rgb_schema = {'varNames': rgb_var_names, 'normRanges': rgb_norm_ranges}
        if client is not None:
            tile_grid = ctx.get_tile_grid(ds_id)
            tile_xyz_source_options = get_tile_source_options(
                tile_grid,
                get_dataset_tile_url(ctx, ds_id,
                                     'rgb',
                                     base_url),
                client=client
            )
            rgb_schema["tileSourceOptions"] = tile_xyz_source_options
        dataset_dict["rgbSchema"] = rgb_schema

    dim_names = ds.data_vars[list(ds.data_vars)[0]].dims \
        if len(ds.data_vars) > 0 else ds.dims.keys()
    dataset_dict["dimensions"] = [
        get_dataset_coordinates(ctx, ds_id, dim_name)
        for dim_name in dim_names
    ]

    dataset_dict["attrs"] = {
        key: ds.attrs[key]
        for key in sorted(list(ds.attrs.keys()))
    }

    dataset_attributions = dataset_config.get(
        'DatasetAttribution',
        ctx.config.get('DatasetAttribution')
    )
    if dataset_attributions is not None:
        if isinstance(dataset_attributions, str):
            dataset_attributions = [dataset_attributions]
        dataset_dict['attributions'] = dataset_attributions

    place_groups = ctx.get_dataset_place_groups(ds_id, base_url)
    if place_groups:
        dataset_dict["placeGroups"] = _filter_place_groups(place_groups,
                                                           del_features=True)

    return dataset_dict


def get_dataset_place_groups(ctx: ServiceContext, ds_id: str, base_url: str) -> List[GeoJsonFeatureCollection]:
    # Do not load or return features, just place group (metadata).
    place_groups = ctx.get_dataset_place_groups(ds_id, base_url, load_features=False)
    return _filter_place_groups(place_groups, del_features=True)


def get_dataset_place_group(ctx: ServiceContext, ds_id: str, place_group_id: str,
                            base_url: str) -> GeoJsonFeatureCollection:
    # Load and return features for specific place group.
    place_group = ctx.get_dataset_place_group(ds_id, place_group_id, base_url, load_features=True)
    return _filter_place_group(place_group, del_features=False)


def get_dataset_coordinates(ctx: ServiceContext, ds_id: str, dim_name: str) -> Dict:
    ds, var = ctx.get_dataset_and_coord_variable(ds_id, dim_name)
    values = list()
    if np.issubdtype(var.dtype, np.floating):
        converter = float
    elif np.issubdtype(var.dtype, np.integer):
        converter = int
    else:
        converter = timestamp_to_iso_string
    for value in var.values:
        values.append(converter(value))
    return dict(name=dim_name,
                size=len(values),
                dtype=str(var.dtype),
                coordinates=values)


# noinspection PyUnusedLocal
def get_color_bars(ctx: ServiceContext, mime_type: str) -> str:
    cmaps = get_cmaps()
    if mime_type == 'application/json':
        return json.dumps(cmaps, indent=2)
    elif mime_type == 'text/html':
        html_head = '<!DOCTYPE html>\n' + \
                    '<html lang="en">\n' + \
                    '<head>' + \
                    '<meta charset="UTF-8">' + \
                    '<title>xcube server color maps</title>' + \
                    '</head>\n' + \
                    '<body style="padding: 0.2em">\n'
        html_body = ''
        html_foot = '</body>\n' \
                    '</html>\n'
        for cmap_cat, cmap_desc, cmap_bars in cmaps:
            html_body += '    <h2>%s</h2>\n' % cmap_cat
            html_body += '    <p><i>%s</i></p>\n' % cmap_desc
            html_body += '    <table style=border: 0">\n'
            for cmap_bar in cmap_bars:
                cmap_name, cmap_data = cmap_bar
                cmap_image = f'<img src="data:image/png;base64,{cmap_data}" width="100%%" height="32"/>'
                html_body += f'        <tr><td style="width: 5em">{cmap_name}:' \
                             f'</td><td style="width: 40em">{cmap_image}</td></tr>\n'
            html_body += '    </table>\n'
        return html_head + html_body + html_foot
    raise ServiceBadRequestError(f'Format {mime_type!r} not supported for color bars')


def _is_point_in_dataset_bbox(point: Tuple[float, float], dataset_dict: Dict):
    if 'bbox' not in dataset_dict:
        return False
    x, y = point
    x_min, y_min, x_max, y_max = dataset_dict['bbox']
    if not (y_min <= y <= y_max):
        return False
    if x_min < x_max:
        return x_min <= x <= x_max
    else:
        # Bounding box crosses antimeridian
        return x_min <= x <= 180.0 or -180.0 <= x <= x_max


def _filter_place_group(place_group: Dict, del_features: bool = False) -> Dict:
    place_group = dict(place_group)
    del place_group['sourcePaths']
    del place_group['sourceEncoding']
    if del_features:
        del place_group['features']
    return place_group


def _filter_place_groups(place_groups, del_features: bool = False) -> List[Dict]:
    if del_features:
        def __filter_place_group(place_group):
            return _filter_place_group(place_group, del_features=True)
    else:
        def __filter_place_group(place_group):
            return _filter_place_group(place_group, del_features=False)

    return list(map(__filter_place_group, place_groups))


class CubeIsNotDisplayable(ValueError):
    pass
