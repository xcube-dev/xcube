import io
import warnings
from typing import Dict, Any

import matplotlib
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.figure

from xcube.core.tile import get_ml_dataset_tile, parse_non_spatial_labels
from xcube.util.cmaps import get_norm, get_cmap
from xcube.util.tilegrid import TileGrid
from xcube.webapi.context import ServiceContext
from xcube.webapi.defaults import DEFAULT_CMAP_WIDTH, DEFAULT_CMAP_HEIGHT
from xcube.webapi.errors import ServiceBadRequestError, ServiceResourceNotFoundError
from xcube.webapi.ne2 import NaturalEarth2Image
from xcube.webapi.reqparams import RequestParams


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
    format = params.get_query_argument('format', 'png')

    ml_dataset = ctx.get_ml_dataset(ds_id)
    if var_name == 'rgb':
        norm_vmin = params.get_query_argument_float('vmin', default=0.0)
        norm_vmax = params.get_query_argument_float('vmax', default=1.0)
        var_names, norm_ranges = ctx.get_rgb_color_mapping(ds_id, norm_range=(norm_vmin, norm_vmax))
        components = ('r', 'g', 'b')
        for i in range(3):
            c = components[i]
            var_names[i] = params.get_query_argument(c, default=var_names[i])
            norm_ranges[i] = params.get_query_argument_float(f'{c}vmin', default=norm_ranges[i][0]), \
                             params.get_query_argument_float(f'{c}vmax', default=norm_ranges[i][1])
        cmap_name = tuple(var_names)
        cmap_range = tuple(norm_ranges)
        for name in var_names:
            if name and name not in ml_dataset.base_dataset:
                raise ServiceBadRequestError(f'Variable {name!r} not found in dataset {ds_id!r}')
        var = None
        for name in var_names:
            if name and name in ml_dataset.base_dataset:
                var = ml_dataset.base_dataset[name]
                break
        if var is None:
            raise ServiceBadRequestError(f'No variable in dataset {ds_id!r} specified for RGB')
    else:
        if format == 'png':
            cmap_name = params.get_query_argument('cbar', default=None)
            cmap_vmin = params.get_query_argument_float('vmin', default=None)
            cmap_vmax = params.get_query_argument_float('vmax', default=None)
            if cmap_name is None or cmap_vmin is None or cmap_vmax is None:
                default_cmap_name, (default_cmap_vmin, default_cmap_vmax) = ctx.get_color_mapping(ds_id, var_name)
                cmap_name = cmap_name or default_cmap_name
                cmap_vmin = cmap_vmin or default_cmap_vmin
                cmap_vmax = cmap_vmax or default_cmap_vmax
            cmap_range = cmap_vmin, cmap_vmax
        elif format == 'raw':
            cmap_name = None
            cmap_range = None
        else:
            raise ServiceBadRequestError(f'Illegal format {format!r}')
        if var_name not in ml_dataset.base_dataset:
            raise ServiceBadRequestError(f'Variable {var_name!r} not found in dataset {ds_id!r}')
        var = ml_dataset.base_dataset[var_name]

    labels = parse_non_spatial_labels(params.get_query_arguments(),
                                      var.dims,
                                      var.coords,
                                      allow_slices=False,
                                      exception_type=ServiceBadRequestError)

    return get_ml_dataset_tile(ml_dataset,
                               var_name,
                               x, y, z,
                               labels=labels,
                               cmap_name=cmap_name,
                               cmap_range=cmap_range,
                               image_cache=ctx.image_cache,
                               tile_cache=ctx.tile_cache,
                               tile_comp_mode=tile_comp_mode,
                               trace_perf=trace_perf,
                               exception_type=ServiceBadRequestError)


def get_legend(ctx: ServiceContext,
               ds_id: str,
               var_name: str,
               params: RequestParams):
    cmap_name = params.get_query_argument('cbar', default=None)
    cmap_vmin = params.get_query_argument_float('vmin', default=None)
    cmap_vmax = params.get_query_argument_float('vmax', default=None)
    cmap_w = params.get_query_argument_int('width', default=None)
    cmap_h = params.get_query_argument_int('height', default=None)
    if cmap_name is None or cmap_vmin is None or cmap_vmax is None or cmap_w is None or cmap_h is None:
        default_cmap_cbar, (default_cmap_vmin, default_cmap_vmax) = ctx.get_color_mapping(ds_id, var_name)
        cmap_name = cmap_name or default_cmap_cbar
        cmap_vmin = cmap_vmin or default_cmap_vmin
        cmap_vmax = cmap_vmax or default_cmap_vmax
        cmap_w = cmap_w or DEFAULT_CMAP_WIDTH
        cmap_h = cmap_h or DEFAULT_CMAP_HEIGHT

    try:
        _, cmap = get_cmap(cmap_name)
    except ValueError:
        raise ServiceResourceNotFoundError(f"color bar {cmap_name!r} not found")

    fig = matplotlib.figure.Figure(figsize=(cmap_w, cmap_h))
    ax1 = fig.add_subplot(1, 1, 1)
    if '.cpd' in cmap_name:
        norm, ticks = get_norm(cmap_name)
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
    if tile_client in ['ol4', 'cesium', 'leaflet']:
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
    elif client == 'cesium':
        # Cesium 1.x
        return _tile_grid_to_cesium1x_source_options(tile_grid, url)
    else:  # client == 'leaflet':
        # Leaflet 1.x
        return _tile_grid_to_leaflet1x_source_options(tile_grid, url)


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

    delta_x = east - west + (0 if east >= west else 360)
    delta_y = north - south
    width = tile_grid.width(0)
    height = tile_grid.height(0)
    res0_x = delta_x / width
    res0_y = delta_y / height
    res0 = max(res0_x, res0_y)
    if abs(res0_y - res0_x) >= 1.e-5:
        warnings.warn(f'spatial resolutions in x and y direction differ significantly:'
                      f' {res0_x} and {res0_y} degrees, using maximum {res0}')

    # https://openlayers.org/en/latest/examples/xyz.html
    # https://openlayers.org/en/latest/apidoc/ol.source.XYZ.html
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

    * https://cesiumjs.org/Cesium/Build/Documentation/UrlTemplateImageryProvider.html

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


def _tile_grid_to_leaflet1x_source_options(tile_grid: TileGrid, url: str):
    """
    Convert TileGrid into options to be used with L.tileLayer(options) of Leaflet 1.7+.

    See

    * https://cesiumjs.org/Cesium/Build/Documentation/UrlTemplateImageryProvider.html

    :param tile_grid: tile grid
    :param url: source url
    :return:
    """
    west, south, east, north = tile_grid.geo_extent
    tile_width = tile_grid.tile_size[0]
    tile_height = tile_grid.tile_size[1]
    return dict(url=url,
                bounds=[west, south, east, north],
                minNativeZoom=0,
                maxNativeZoom=tile_grid.num_levels - 1,
                tileSize=tile_width if tile_width == tile_height else [tile_width, tile_height])
