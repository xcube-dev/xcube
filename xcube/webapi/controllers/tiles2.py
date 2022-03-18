import logging

from xcube.core.tile2 import DEFAULT_CMAP_NAME
from xcube.core.tile2 import DEFAULT_CRS_NAME
from xcube.core.tile2 import DEFAULT_FORMAT
from xcube.core.tile2 import compute_rgba_tile
from xcube.util.tilegrid2 import DEFAULT_TILE_SIZE
from xcube.webapi.context import ServiceContext
from xcube.webapi.errors import ServiceBadRequestError
from xcube.webapi.reqparams import RequestParams

_LOGGER = logging.getLogger()


def get_dataset_tile2(ctx: ServiceContext,
                      ds_id: str,
                      var_name: str,
                      x: str, y: str, z: str,
                      params: RequestParams):
    x = RequestParams.to_int('x', x)
    y = RequestParams.to_int('y', y)
    z = RequestParams.to_int('z', z)

    args = dict(params.get_query_arguments())

    log_tiles = args.pop('debug', None) == '1' or ctx.trace_perf
    format = args.pop('format', DEFAULT_FORMAT)
    crs_name = args.pop('crs', DEFAULT_CRS_NAME)
    retina = args.pop('retina', None) == '1'
    cmap_name = args.pop('cbar', DEFAULT_CMAP_NAME)
    cmap_name = args.pop('cmap', cmap_name)
    cmap_vmin = float(args.pop('vmin', 0.0))
    cmap_vmax = float(args.pop('vmax', 1.0))

    ml_dataset = ctx.get_ml_dataset(ds_id)
    if var_name == 'rgb':
        norm_vmin = cmap_vmin
        norm_vmax = cmap_vmax
        var_names, norm_ranges = ctx.get_rgb_color_mapping(
            ds_id, norm_range=(norm_vmin, norm_vmax)
        )
        components = 'r', 'g', 'b'
        for i, c in enumerate(components):
            var_names[i] = args.pop(c, var_names[i])
            norm_ranges[i] = (
                float(args.pop(
                    f'{c}vmin', norm_ranges[i][0]
                )),
                float(args.pop(
                    f'{c}vmax', norm_ranges[i][1]
                ))
            )
        cmap_name = tuple(var_names)
        cmap_range = tuple(norm_ranges)
        for name in var_names:
            if name and name not in ml_dataset.base_dataset:
                raise ServiceBadRequestError(
                    f'Variable {name!r} not found in dataset {ds_id!r}'
                )
        var = None
        for name in var_names:
            if name and name in ml_dataset.base_dataset:
                var = ml_dataset.base_dataset[name]
                break
        if var is None:
            raise ServiceBadRequestError(
                f'No variable in dataset {ds_id!r} specified for RGB'
            )
    else:
        if format == 'png' or format == 'image/png':
            if cmap_name is None or cmap_vmin is None or cmap_vmax is None:
                default_cmap_name, (default_cmap_vmin, default_cmap_vmax) = \
                    ctx.get_color_mapping(ds_id, var_name)
                if cmap_name is None:
                    cmap_name = default_cmap_name
                if cmap_vmin is None:
                    cmap_vmin = default_cmap_vmin
                if cmap_vmax is None:
                    cmap_vmax = default_cmap_vmax
            cmap_range = cmap_vmin, cmap_vmax
        elif format == 'raw' or format == 'image/raw':
            cmap_name = None
            cmap_range = None
        else:
            raise ServiceBadRequestError(
                f'Illegal format {format!r}'
            )
        if var_name not in ml_dataset.base_dataset:
            raise ServiceBadRequestError(
                f'Variable {var_name!r} not found in dataset {ds_id!r}'
            )

    return compute_rgba_tile(
        ml_dataset,
        var_name,
        x, y, z,
        crs_name=crs_name,
        tile_size=(2 if retina else 1) * DEFAULT_TILE_SIZE,
        cmap_name=cmap_name,
        value_range=cmap_range,
        non_spatial_labels=args,
        format=format,
        logger=_LOGGER if log_tiles else None,
    )
