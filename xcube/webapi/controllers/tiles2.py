import logging
from typing import Optional

from xcube.core.tile2 import DEFAULT_CRS_NAME
from xcube.core.tile2 import DEFAULT_FORMAT
from xcube.core.tile2 import TileNotFoundException
from xcube.core.tile2 import TileRequestException
from xcube.core.tile2 import compute_rgba_tile
from xcube.util.tilegrid2 import DEFAULT_TILE_SIZE
from xcube.webapi.context import ServiceContext
from xcube.webapi.errors import ServiceBadRequestError
from xcube.webapi.errors import ServiceResourceNotFoundError
from xcube.webapi.reqparams import RequestParams

_LOGGER = logging.getLogger()


def get_dataset_tile2(ctx: ServiceContext,
                      ds_id: str,
                      var_name: str,
                      crs_name: Optional[str],
                      x: str, y: str, z: str,
                      params: RequestParams):
    x = RequestParams.to_int('x', x)
    y = RequestParams.to_int('y', y)
    z = RequestParams.to_int('z', z)

    args = dict(params.get_query_arguments())

    crs_name = args.pop('crs', crs_name or DEFAULT_CRS_NAME)
    retina = args.pop('retina', None) == '1'
    cmap_name = args.pop('cmap', args.pop('cbar', None))
    value_min = float(args.pop('vmin', 0.0))
    value_max = float(args.pop('vmax', 1.0))
    format = args.pop('format', DEFAULT_FORMAT)
    log_tiles = args.pop('debug', None) == '1' or ctx.trace_perf

    if format not in ('png' or 'image/png'):
        raise ServiceBadRequestError(
            f'Illegal format {format!r}'
        )

    ml_dataset = ctx.get_ml_dataset(ds_id)
    if var_name == 'rgb':
        var_names, value_ranges = ctx.get_rgb_color_mapping(
            ds_id, norm_range=(value_min, value_max)
        )
        components = 'r', 'g', 'b'
        for i, c in enumerate(components):
            var_names[i] = args.pop(c, var_names[i])
            value_ranges[i] = (
                float(args.pop(
                    f'{c}vmin', value_ranges[i][0]
                )),
                float(args.pop(
                    f'{c}vmax', value_ranges[i][1]
                ))
            )
    else:
        if cmap_name is None or value_min is None or value_max is None:
            default_cmap_name, (default_value_min, default_value_min) = \
                ctx.get_color_mapping(ds_id, var_name)
            if cmap_name is None:
                cmap_name = default_cmap_name
            if value_min is None:
                value_min = default_value_min
            if value_max is None:
                value_max = default_value_min
        var_names = (var_name,)
        value_ranges = ((value_min, value_max),)

    try:
        return compute_rgba_tile(
            ml_dataset,
            var_names,
            x, y, z,
            crs_name=crs_name,
            tile_size=(2 if retina else 1) * DEFAULT_TILE_SIZE,
            cmap_name=cmap_name,
            value_ranges=value_ranges,
            non_spatial_labels=args,
            format=format,
            logger=_LOGGER if log_tiles else None,
        )
    except TileNotFoundException as e:
        raise ServiceResourceNotFoundError(f'{e}') from e
    except TileRequestException as e:
        raise ServiceBadRequestError(f'{e}') from e
