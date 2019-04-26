import json
from typing import Dict

import numpy as np

from ..context import ServiceContext
from ..controllers.tiles import get_tile_source_options, get_dataset_tile_url
from ..errors import ServiceBadRequestError
from ..im.cmaps import get_cmaps
from ..utils import get_dataset_bounds, timestamp_to_iso_string


def get_datasets(ctx: ServiceContext, details=False, client=None, base_url: str = None) -> Dict:
    dataset_descriptors = ctx.get_dataset_descriptors()

    dataset_dicts = list()
    for dataset_descriptor in dataset_descriptors:
        ds_id = dataset_descriptor['Identifier']
        dataset_dict = dict(id=ds_id)

        if 'Title' in dataset_descriptor:
            ds_title = dataset_descriptor['Title']
            if ds_title and isinstance(ds_title, str):
                dataset_dict['title'] = ds_title
            else:
                dataset_dict['title'] = ds_id

        if 'BoundingBox' in dataset_descriptor:
            ds_bbox = dataset_descriptor['BoundingBox']
            if ds_bbox \
                    and len(ds_bbox) == 4 \
                    and all(map(lambda c: isinstance(c, float) or isinstance(c, int), ds_bbox)):
                dataset_dict['bbox'] = ds_bbox

        dataset_dicts.append(dataset_dict)

    if details:
        for dataset_dict in dataset_dicts:
            ds_id = dataset_dict["id"]
            dataset_dict.update(get_dataset(ctx, ds_id, client, base_url))

    return dict(datasets=dataset_dicts)


def get_dataset(ctx: ServiceContext, ds_id: str, client=None, base_url: str = None) -> Dict:
    dataset_descriptor = ctx.get_dataset_descriptor(ds_id)

    ds_id = dataset_descriptor['Identifier']
    ds_title = dataset_descriptor['Title']
    dataset_dict = dict(id=ds_id, title=ds_title)

    ds = ctx.get_dataset(ds_id)
    dataset_dict["bbox"] = list(get_dataset_bounds(ds))

    variable_dicts = []
    for var_name in ds.data_vars:
        var = ds.data_vars[var_name]
        dims = var.dims
        if len(dims) < 3 or dims[0] != 'time' or dims[-2] != 'lat' or dims[-1] != 'lon':
            continue

        variable_dict = dict(id=f'{ds_id}.{var_name}',
                             name=var_name,
                             dims=list(dims),
                             shape=list(var.shape),
                             dtype=str(var.dtype),
                             units=var.attrs.get('units', ''),
                             title=var.attrs.get('title', var.attrs.get('long_name', var_name)))

        if client is not None:
            tile_grid = ctx.get_tile_grid(ds_id)
            tile_xyz_source_options = get_tile_source_options(tile_grid,
                                                              get_dataset_tile_url(ctx, ds_id, var_name,
                                                                                   base_url),
                                                              client=client)
            variable_dict["tileSourceOptions"] = tile_xyz_source_options

        cbar, vmin, vmax = ctx.get_color_mapping(ds_id, var_name)
        variable_dict["colorBarName"] = cbar
        variable_dict["colorBarMin"] = vmin
        variable_dict["colorBarMax"] = vmax

        variable_dicts.append(variable_dict)

    dataset_dict["variables"] = variable_dicts

    dim_names = ds.data_vars[list(ds.data_vars)[0]].dims if len(ds.data_vars) > 0 else ds.dims.keys()
    dataset_dict["dimensions"] = [get_dataset_coordinates(ctx, ds_id, dim_name) for dim_name in dim_names]

    place_groups = ctx.get_dataset_place_groups(ds_id)
    if place_groups:
        dataset_dict["placeGroups"] = place_groups

    return dataset_dict


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
