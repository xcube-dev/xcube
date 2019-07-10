import io
import logging
from typing import Dict, Any

import matplotlib
import matplotlib.cm as cm
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.figure
import numpy as np

from xcube.webapi.im.cmaps import get_norm
from ..context import ServiceContext
from ..defaults import DEFAULT_CMAP_WIDTH, DEFAULT_CMAP_HEIGHT
from ..errors import ServiceBadRequestError, ServiceResourceNotFoundError
from ..im import NdarrayImage, TransformArrayImage, ColorMappedRgbaImage, ColorMappedRgbaImage2, TileGrid
from ..ne2 import NaturalEarth2Image
from ..reqparams import RequestParams
from ...util.perf import measure_time_cm

_LOG = logging.getLogger('xcube')


def get_dataset_tile(ctx: ServiceContext,
                     ds_id: str,
                     var_name: str,
                     x: str, y: str, z: str,
                     params: RequestParams):
    x = RequestParams.to_int('x', x)
    y = RequestParams.to_int('y', y)
    z = RequestParams.to_int('z', z)

    tile_comp_mode = params.get_query_argument_int('mode', ctx.tile_comp_mode)
    trace_perf = params.get_query_argument_int('debug', ctx.trace_perf) != 0

    measure_time = measure_time_cm(logger=_LOG, disabled=not trace_perf)

    var = ctx.get_variable_for_z(ds_id, var_name, z)

    dim_names = list(var.dims)
    if 'lon' not in dim_names or 'lat' not in dim_names:
        raise ServiceBadRequestError(f'Variable "{var_name}" of dataset "{ds_id}" is not geo-spatial')

    dim_names.remove('lon')
    dim_names.remove('lat')

    var_indexers = ctx.get_var_indexers(ds_id, var_name, var, dim_names, params)

    cmap_cbar = params.get_query_argument('cbar', default=None)
    cmap_vmin = params.get_query_argument_float('vmin', default=None)
    cmap_vmax = params.get_query_argument_float('vmax', default=None)
    if cmap_cbar is None or cmap_vmin is None or cmap_vmax is None:
        default_cmap_cbar, default_cmap_vmin, default_cmap_vmax = ctx.get_color_mapping(ds_id, var_name)
        cmap_cbar = cmap_cbar or default_cmap_cbar
        cmap_vmin = cmap_vmin or default_cmap_vmin
        cmap_vmax = cmap_vmax or default_cmap_vmax

    image_id = '-'.join([ds_id, f"{z}", var_name]
                        + [f'{dim_name}={dim_value}' for dim_name, dim_value in var_indexers.items()])

    if image_id in ctx.image_cache:
        image = ctx.image_cache[image_id]
    else:
        no_data_value = var.attrs.get('_FillValue')
        valid_range = var.attrs.get('valid_range')
        if valid_range is None:
            valid_min = var.attrs.get('valid_min')
            valid_max = var.attrs.get('valid_max')
            if valid_min is not None and valid_max is not None:
                valid_range = [valid_min, valid_max]

        # Make sure we work with 2D image arrays only
        if var.ndim == 2:
            assert len(var_indexers) == 0
            array = var
        elif var.ndim > 2:
            assert len(var_indexers) == var.ndim - 2
            array = var.sel(method='nearest', **var_indexers)
        else:
            raise ServiceBadRequestError(f'Variable "{var_name}" of dataset "{var_name}" '
                                         'must be an N-D Dataset with N >= 2, '
                                         f'but "{var_name}" is only {var.ndim}-D')

        cmap_vmin = np.nanmin(array.values) if np.isnan(cmap_vmin) else cmap_vmin
        cmap_vmax = np.nanmax(array.values) if np.isnan(cmap_vmax) else cmap_vmax

        tile_grid = ctx.get_tile_grid(ds_id)

        if not tile_comp_mode:
            image = NdarrayImage(array,
                                 image_id=f'ndai-{image_id}',
                                 tile_size=tile_grid.tile_size,
                                 # tile_cache=ctx.tile_cache,
                                 trace_perf=trace_perf)
            image = TransformArrayImage(image,
                                        image_id=f'tai-{image_id}',
                                        flip_y=tile_grid.inv_y,
                                        force_masked=True,
                                        no_data_value=no_data_value,
                                        valid_range=valid_range,
                                        # tile_cache=ctx.tile_cache,
                                        trace_perf=trace_perf)
            image = ColorMappedRgbaImage(image,
                                         image_id=f'rgb-{image_id}',
                                         value_range=(cmap_vmin, cmap_vmax),
                                         cmap_name=cmap_cbar,
                                         encode=True,
                                         format='PNG',
                                         tile_cache=ctx.tile_cache,
                                         trace_perf=trace_perf)
        else:
            image = ColorMappedRgbaImage2(array,
                                          image_id=f'rgb-{image_id}',
                                          tile_size=tile_grid.tile_size,
                                          cmap_range=(cmap_vmin, cmap_vmax),
                                          cmap_name=cmap_cbar,
                                          encode=True,
                                          format='PNG',
                                          flip_y=tile_grid.inv_y,
                                          no_data_value=no_data_value,
                                          valid_range=valid_range,
                                          tile_cache=ctx.tile_cache,
                                          trace_perf=trace_perf)

        ctx.image_cache[image_id] = image
        if trace_perf:
            _LOG.info(f'Created tiled image {image_id!r} of size {image.size} with tile grid:')
            _LOG.info(f'  num_levels: {tile_grid.num_levels}')
            _LOG.info(f'  num_level_zero_tiles: {tile_grid.num_tiles(0)}')
            _LOG.info(f'  tile_size: {tile_grid.tile_size}')
            _LOG.info(f'  geo_extent: {tile_grid.geo_extent}')
            _LOG.info(f'  inv_y: {tile_grid.inv_y}')

    if trace_perf:
        _LOG.info(f'>>> tile {image_id}/{z}/{y}/{x}')

    with measure_time() as measured_time:
        tile = image.get_tile(x, y)

    if trace_perf:
        _LOG.info(f'<<< tile {image_id}/{z}/{y}/{x}: took ' + '%.2f seconds' % measured_time.duration)

    return tile


def get_legend(ctx: ServiceContext,
               ds_id: str,
               var_name: str,
               params: RequestParams):
    cmap_cbar = params.get_query_argument('cbar', default=None)
    cmap_vmin = params.get_query_argument_float('vmin', default=None)
    cmap_vmax = params.get_query_argument_float('vmax', default=None)
    cmap_w = params.get_query_argument_int('width', default=None)
    cmap_h = params.get_query_argument_int('height', default=None)
    if cmap_cbar is None or cmap_vmin is None or cmap_vmax is None or cmap_w is None or cmap_h is None:
        default_cmap_cbar, default_cmap_vmin, default_cmap_vmax = ctx.get_color_mapping(ds_id, var_name)
        cmap_cbar = cmap_cbar or default_cmap_cbar
        cmap_vmin = cmap_vmin or default_cmap_vmin
        cmap_vmax = cmap_vmax or default_cmap_vmax
        cmap_w = cmap_w or DEFAULT_CMAP_WIDTH
        cmap_h = cmap_h or DEFAULT_CMAP_HEIGHT

    try:
        cmap = cm.get_cmap(cmap_cbar)
    except ValueError:
        raise ServiceResourceNotFoundError(f"color bar {cmap_cbar} not found")

    fig = matplotlib.figure.Figure(figsize=(cmap_w, cmap_h))
    ax1 = fig.add_subplot(1, 1, 1)
    if '.cpd' in cmap_cbar:
        norm, ticks = get_norm(cmap_cbar)
    else:
        norm = matplotlib.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
        ticks = None

    image_legend = matplotlib.colorbar.ColorbarBase(ax1,
                                                    format='%.1f',
                                                    ticks=ticks,
                                                    cmap=cmap,
                                                    norm=norm,
                                                    orientation='vertical')

    image_legend_label = ctx.get_legend_label(ds_id, var_name)
    if image_legend_label is not None:
        image_legend.set_label(image_legend_label)

    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.0)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')

    return buffer.getvalue()


def get_dataset_tile_grid(ctx: ServiceContext,
                          ds_id: str,
                          var_name: str,
                          tile_client: str,
                          base_url: str) -> Dict[str, Any]:
    tile_grid = ctx.get_tile_grid(ds_id)
    if tile_client == 'ol4' or tile_client == 'cesium':
        return get_tile_source_options(tile_grid,
                                       get_dataset_tile_url(ctx, ds_id, var_name, base_url),
                                       client=tile_client)
    else:
        raise ServiceBadRequestError(f'Unknown tile client "{tile_client}"')


def get_dataset_tile_url(ctx: ServiceContext, ds_id: str, var_name: str, base_url: str):
    return ctx.get_service_url(base_url, 'datasets', ds_id, 'vars', var_name, 'tiles', '{z}/{x}/{y}.png')


# noinspection PyUnusedLocal
def get_ne2_tile(ctx: ServiceContext, x: str, y: str, z: str, params: RequestParams):
    x = params.to_int('x', x)
    y = params.to_int('y', y)
    z = params.to_int('z', z)
    return NaturalEarth2Image.get_pyramid().get_tile(x, y, z)


def get_ne2_tile_grid(ctx: ServiceContext, tile_client: str, base_url: str):
    if tile_client == 'ol4':
        return get_tile_source_options(NaturalEarth2Image.get_pyramid().tile_grid,
                                       get_ne2_tile_url(ctx, base_url),
                                       client=tile_client)
    else:
        raise ServiceBadRequestError(f'Unknown tile client {tile_client!r}')


def get_ne2_tile_url(ctx: ServiceContext, base_url: str):
    return ctx.get_service_url(base_url, 'ne2', 'tiles', '{z}/{x}/{y}.jpg')


def get_tile_source_options(tile_grid: TileGrid, url: str, client: str = 'ol4'):
    if client == 'ol4':
        # OpenLayers 4.x
        return _tile_grid_to_ol4x_xyz_source_options(tile_grid, url)
    else:
        # Cesium 1.x
        return _tile_grid_to_cesium1x_source_options(tile_grid, url)


def _tile_grid_to_ol4x_xyz_source_options(tile_grid: TileGrid, url: str):
    """
    Convert TileGrid into options to be used with ol.source.XYZ(options) of OpenLayers 4.x.

    See

    * https://openlayers.org/en/latest/apidoc/ol.source.XYZ.html
    * https://openlayers.org/en/latest/examples/xyz.html

    :param tile_grid: tile grid
    :param url: source url
    :return:
    """
    west, south, east, north = tile_grid.geo_extent
    res0 = (north - south) / tile_grid.height(0)
    #   https://openlayers.org/en/latest/examples/xyz.html
    #   https://openlayers.org/en/latest/apidoc/ol.source.XYZ.html
    return dict(url=url,
                projection='EPSG:4326',
                minZoom=0,
                maxZoom=tile_grid.num_levels - 1,
                tileGrid=dict(extent=[west, south, east, north],
                              origin=[west, south if tile_grid.inv_y else north],
                              tileSize=[tile_grid.tile_size[0], tile_grid.tile_size[1]],
                              resolutions=[res0 / (2 ** i) for i in range(tile_grid.num_levels)]))


def _tile_grid_to_cesium1x_source_options(tile_grid: TileGrid, url: str):
    """
    Convert TileGrid into options to be used with Cesium.UrlTemplateImageryProvider(options) of Cesium 1.45+.

    See

    * https://cesiumjs.org/Cesium/Build/Documentation/UrlTemplateImageryProvider.html?classFilter=UrlTemplateImageryProvider

    :param tile_grid: tile grid
    :param url: source url
    :return:
    """
    west, south, east, north = tile_grid.geo_extent
    rectangle = dict(west=west, south=south, east=east, north=north)
    return dict(url=url,
                rectangle=rectangle,
                minimumLevel=0,
                maximumLevel=tile_grid.num_levels - 1,
                tileWidth=tile_grid.tile_size[0],
                tileHeight=tile_grid.tile_size[1],
                tilingScheme=dict(rectangle=rectangle,
                                  numberOfLevelZeroTilesX=tile_grid.num_level_zero_tiles_x,
                                  numberOfLevelZeroTilesY=tile_grid.num_level_zero_tiles_y))
