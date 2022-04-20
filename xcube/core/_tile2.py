# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import io
import logging
import math
import warnings
from typing import Optional, Tuple, Dict, Any, Hashable, Union, Sequence

import PIL
import matplotlib.colors
import numpy as np
import xarray as xr

from .mldataset import MultiLevelDataset
from .tilingscheme import DEFAULT_CRS_NAME
from .tilingscheme import DEFAULT_TILE_SIZE
from .tilingscheme import TilingScheme
from ..constants import LOG
from ..util.assertions import assert_in
from ..util.assertions import assert_instance
from ..util.assertions import assert_true
from ..util.perf import measure_time_cm
from ..util.projcache import ProjCache

DEFAULT_VALUE_RANGE = (0., 1.)
DEFAULT_CMAP_NAME = 'bone'
DEFAULT_FORMAT = 'png'
DEFAULT_TILE_ENLARGEMENT = 1

_ALMOST_256 = 256 - 1e-10

ValueRange = Tuple[float, float]


def compute_rgba_tile(
        ml_dataset: MultiLevelDataset,
        variable_names: Union[str, Sequence[str]],
        tile_x: int,
        tile_y: int,
        tile_z: int,
        crs_name: str = DEFAULT_CRS_NAME,
        tile_size: int = DEFAULT_TILE_SIZE,
        cmap_name: str = None,
        value_ranges: Optional[Union[ValueRange,
                                     Sequence[ValueRange]]] = None,
        non_spatial_labels: Optional[Dict[str, Any]] = None,
        format: str = DEFAULT_FORMAT,
        tile_enlargement: int = DEFAULT_TILE_ENLARGEMENT,
        trace_perf: bool = False
) -> Union[bytes, np.ndarray]:
    """Compute an RGBA image tile from *variable_names* in
    given multi-resolution dataset *mr_dataset*.

    If length of *variable_names* is three, we create a direct component
    RGBA image where the three variables correspond to the three R, G, B
    channels and A is computed from NaN values in the three variables.
    *value_ranges* must have exactly three elements too.

    If length of *variable_names* is one, we create a color mapped
    RGBA image using *cmap_name* where R, G, B is computed from a color bar
    and A is computed from NaN values in the variable.
    *value_ranges* must have exactly one element.

    The algorithm is as follows:

    1. compute the z-index of the dataset pyramid from requested
       map CRS and tile_z.
    2. select a spatial 2D slice from the dataset pyramid.
    3. create a 2D array of the tile's map coordinates at each pixel
    4. transform 2D array of tile map coordinates into coordinates of
       the dataset CRS.
    5. use the 2D array's outline to retrieve the bbox in dataset
       coordinates.
    6. spatially subset the 2D slice using that bbox.
    7. compute a new bbox from the actual spatial subset.
    8. use the new bbox to transform the 2D array of the tile's dataset
       coordinates into indexes into the spatial subset.
    9. use the tile indices as 2D index into the spatial subset.
       The result is a reprojected array with the shape of the tile.
    10. turn that array into an RGBA image.
    11. encode RGBA image into PNG bytes.

    :param ml_dataset: Multi-level dataset
    :param variable_names: Single variable name
        or a sequence of three names.
    :param tile_x: Tile X coordinate
    :param tile_y: Tile Y coordinate
    :param tile_z:  Tile Z coordinate
    :param crs_name: Spatial tile coordinate reference system.
        Must be a geographical CRS, such as "EPSG:4326", or
        web mercator, i.e. "EPSG:3857". Defaults to "CRS84".
    :param tile_size: The tile size in pixels. Defaults to 256.
    :param cmap_name: Color map name. Only used if a single
        variable name is given. Defaults to "bone".
    :param value_ranges: A single value range, or value ranges
        for each variable name.
    :param non_spatial_labels: Labels for the non-spatial dimensions
        in the given variables.
    :param tile_enlargement: Enlargement in pixels applied to
        the computed source tile read from the data.
        Can be used to increase the accuracy of the borders of target
        tiles at high zoom levels. Defaults to 1.
    :param format: Either 'png', 'image/png' or 'numpy'.
    :param trace_perf: If set, detailed performance
        metrics are logged using the level of the "xcube" logger.
    :return: PNG bytes or unit8 numpy array, depending on *format*
    :raise TileNotFoundException
    :raise TileRequestException
    """
    if isinstance(variable_names, str):
        variable_names = (variable_names,)
    num_components = len(variable_names)
    assert_true(num_components in (1, 3),
                message='number of names in'
                        ' variable_names must be 1 or 3')
    if not value_ranges:
        value_ranges = num_components * (DEFAULT_VALUE_RANGE,)
    else:
        assert_instance(value_ranges, (list, tuple), name='value_ranges')
    if isinstance(value_ranges[0], (int, float)):
        value_ranges = num_components * (value_ranges,)
    assert_true(num_components == len(value_ranges),
                message='value_ranges must have'
                        ' same length as variable_names')
    format = _normalize_format(format)
    assert_in(format, ('png', 'numpy'), name='format')

    logger = LOG if trace_perf else None

    measure_time = measure_time_cm(disabled=not trace_perf, logger=LOG)

    with measure_time('Preparing 2D subset'):
        tiling_scheme = TilingScheme.for_crs(crs_name) \
            .derive(tile_size=tile_size)

        ds_level = tiling_scheme.get_resolutions_level(
            tile_z,
            ml_dataset.avg_resolutions,
            ml_dataset.grid_mapping.spatial_unit_name
        )
        dataset = ml_dataset.get_dataset(ds_level)

        variables = [
            _get_variable(ml_dataset.ds_id,
                          dataset,
                          variable_name,
                          non_spatial_labels,
                          logger)
            for variable_name in variable_names
        ]

    variable_0 = variables[0]

    with measure_time('Transforming tile map to dataset coordinates'):
        ds_x_name, ds_y_name = ml_dataset.grid_mapping.xy_dim_names

        ds_y_coords = variable_0[ds_y_name]
        ds_y_points_up = bool(ds_y_coords[0] < ds_y_coords[-1])

        tile_bbox = tiling_scheme.get_tile_extent(tile_x, tile_y, tile_z)
        if tile_bbox is None:
            raise TileRequestException.new(
                'tile indices out of tile grid bounds',
                logger=logger
            )
        tile_x_min, tile_y_min, tile_x_max, tile_y_max = tile_bbox

        tile_res = (tile_x_max - tile_x_min) / (tile_size - 1)
        tile_res05 = tile_res / 2

        tile_x_1d = np.linspace(tile_x_min + tile_res05,
                                tile_x_max - tile_res05, tile_size)
        tile_y_1d = np.linspace(tile_y_min + tile_res05,
                                tile_y_max - tile_res05, tile_size)

        tile_x_2d = np.tile(tile_x_1d, (tile_size, 1))
        tile_y_2d = np.tile(tile_y_1d, (tile_size, 1)).transpose()

        t_map_to_ds = ProjCache.INSTANCE.get_transformer(
            tiling_scheme.crs,
            ml_dataset.grid_mapping.crs
        )

        tile_ds_x_2d, tile_ds_y_2d = t_map_to_ds.transform(tile_x_2d,
                                                           tile_y_2d)

    with measure_time('Getting spatial subset'):

        # Get min/max of the 1D arrays surrounding the 2D array
        # North
        ds_x_n = tile_ds_x_2d[0, :]
        ds_y_n = tile_ds_y_2d[0, :]
        # South
        ds_x_s = tile_ds_x_2d[tile_size - 1, :]
        ds_y_s = tile_ds_y_2d[tile_size - 1, :]
        # West
        ds_x_w = tile_ds_x_2d[:, 0]
        ds_y_w = tile_ds_y_2d[:, 0]
        # East
        ds_x_e = tile_ds_x_2d[:, tile_size - 1]
        ds_y_e = tile_ds_y_2d[:, tile_size - 1]
        # Min
        ds_x_min = np.nanmin([np.nanmin(ds_x_n), np.nanmin(ds_x_s),
                              np.nanmin(ds_x_w), np.nanmin(ds_x_e)])
        ds_y_min = np.nanmin([np.nanmin(ds_y_n), np.nanmin(ds_y_s),
                              np.nanmin(ds_y_w), np.nanmin(ds_y_e)])
        # Max
        ds_x_max = np.nanmax([np.nanmax(ds_x_n), np.nanmax(ds_x_s),
                              np.nanmax(ds_x_w), np.nanmax(ds_x_e)])
        ds_y_max = np.nanmax([np.nanmax(ds_y_n), np.nanmax(ds_y_s),
                              np.nanmax(ds_y_w), np.nanmax(ds_y_e)])
        if np.isnan(ds_x_min) or np.isnan(ds_y_min) \
                or np.isnan(ds_y_max) or np.isnan(ds_y_max):
            raise TileNotFoundException.new(
                'tile bounds NaN after map projection',
                logger=logger
            )

        num_extra_pixels = tile_enlargement
        res_x = (ds_x_max - ds_x_min) / tile_size
        res_y = (ds_y_max - ds_y_min) / tile_size
        extra_dx = num_extra_pixels * res_x
        extra_dy = num_extra_pixels * res_y
        ds_x_slice = slice(ds_x_min - extra_dx, ds_x_max + extra_dx)
        if ds_y_points_up:
            ds_y_slice = slice(ds_y_min - extra_dy, ds_y_max + extra_dy)
        else:
            ds_y_slice = slice(ds_y_max + extra_dy, ds_y_min - extra_dy)

        var_subsets = [variable.sel({ds_x_name: ds_x_slice,
                                     ds_y_name: ds_y_slice})
                       for variable in variables]
        for var_subset in var_subsets:
            # A zero or a one in the tile's shape will produce a
            # non-existing or too small image. It will also prevent
            # determining the current resolution.
            if 0 in var_subset.shape or 1 in var_subset.shape:
                return TransparentRgbaTilePool.INSTANCE.get(tile_size, format)

    with measure_time('Transforming dataset coordinates into indices'):
        var_subset_0 = var_subsets[0]
        ds_x_coords = var_subset_0[ds_x_name]
        ds_y_coords = var_subset_0[ds_y_name]

        ds_x1 = float(ds_x_coords[0])
        ds_x2 = float(ds_x_coords[-1])
        ds_y1 = float(ds_y_coords[0])
        ds_y2 = float(ds_y_coords[-1])

        ds_size_x = ds_x_coords.size
        ds_size_y = ds_y_coords.size

        ds_dx = (ds_x2 - ds_x1) / (ds_size_x - 1)
        ds_dy = (ds_y2 - ds_y1) / (ds_size_y - 1)

        ds_x_indices = (tile_ds_x_2d - ds_x1) / ds_dx
        ds_y_indices = (tile_ds_y_2d - ds_y1) / ds_dy

        ds_x_indices = ds_x_indices.astype(dtype=np.int64)
        ds_y_indices = ds_y_indices.astype(dtype=np.int64)

    with measure_time('Masking dataset indices'):
        ds_mask = (ds_x_indices >= 0) & (ds_x_indices < ds_size_x) \
                  & (ds_y_indices >= 0) & (ds_y_indices < ds_size_y)

        ds_x_indices = np.where(ds_mask, ds_x_indices, 0)
        ds_y_indices = np.where(ds_mask, ds_y_indices, 0)

    var_tiles = []
    for var_subset, value_range in zip(var_subsets, value_ranges):
        with measure_time('Loading 2D data for spatial subset'):
            # Note, we need to load the values here into a numpy array,
            # because 2D indexing by [ds_y_indices, ds_x_indices]
            # does not (yet) work with dask arrays.
            var_tile = var_subset.values
            # Remove any axes above the 2nd. This is safe,
            # they will be of size one, if any.
            var_tile = var_tile.reshape(var_tile.shape[-2:])

        with measure_time('Looking up dataset indices'):
            # This does the actual projection trick.
            # Lookup indices ds_y_indices, ds_x_indices to create
            # the actual tile.
            var_tile = var_tile[ds_y_indices, ds_x_indices]
            var_tile = np.where(ds_mask, var_tile, np.nan)

        with measure_time('Normalizing data tile'):
            var_tile = var_tile[::-1, :]
            value_min, value_max = value_range
            if value_max < value_min:
                value_min, value_max = value_max, value_min
            if math.isclose(value_min, value_max):
                value_max = value_min + 1
            norm = matplotlib.colors.Normalize(value_min, value_max)
            var_tile_norm = norm(var_tile)

        var_tiles.append(var_tile_norm)

    with measure_time('Encoding tile as RGBA image'):
        if len(var_tiles) == 1:
            var_tile_norm = var_tiles[0]
            cm = matplotlib.cm.get_cmap(cmap_name or DEFAULT_CMAP_NAME)
            var_tile_rgba = cm(var_tile_norm)
            var_tile_rgba = (255 * var_tile_rgba).astype(np.uint8)
        else:
            r, g, b = var_tiles
            var_tile_rgba = np.zeros((tile_size, tile_size, 4),
                                     dtype=np.uint8)
            var_tile_rgba[..., 0] = _ALMOST_256 * r
            var_tile_rgba[..., 1] = _ALMOST_256 * g
            var_tile_rgba[..., 2] = _ALMOST_256 * b
            var_tile_rgba[..., 3] = np.where(np.isfinite(r + g + b), 255, 0)

    if format == 'png':
        with measure_time('Encoding RGBA image as PNG bytes'):
            return _encode_rgba_as_png(var_tile_rgba)
    else:
        return var_tile_rgba


def _get_variable(ds_name,
                  dataset,
                  variable_name,
                  non_spatial_labels,
                  logger):
    if variable_name not in dataset:
        raise TileNotFoundException.new(
            f'variable {variable_name!r}'
            f' not found in {ds_name!r}',
            logger=logger
        )
    variable = dataset[variable_name]
    non_spatial_labels = _get_non_spatial_labels(dataset,
                                                 variable,
                                                 non_spatial_labels,
                                                 logger)
    if non_spatial_labels:
        variable = variable.sel(**non_spatial_labels, method='nearest')
    return variable


class TileException(Exception):
    @classmethod
    def new(cls,
            message: str,
            logger: Optional[logging.Logger] = None) -> 'TileException':
        if logger is not None:
            logger.warning(message)
        return cls(message)


class TileNotFoundException(TileException):
    pass


class TileRequestException(TileException):
    pass


class TransparentRgbaTilePool:
    """A cache for fully-transparent RGBA tiles of a given size and format."""

    INSTANCE: 'TransparentRgbaTilePool'

    def __init__(self):
        self._transparent_tiles: Dict[str, Union[bytes, np.ndarray]] = dict()

    def get(self, tile_size: int, format: str) -> Union[bytes, np.ndarray]:
        key = f'{format}-{tile_size}'
        if key not in self._transparent_tiles:
            data = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
            if format == 'png':
                data = _encode_rgba_as_png(data)
            self._transparent_tiles[key] = data
        return self._transparent_tiles[key]


TransparentRgbaTilePool.INSTANCE = TransparentRgbaTilePool()


def _get_non_spatial_labels(dataset: xr.Dataset,
                            variable: xr.DataArray,
                            labels: Optional[Dict[str, Any]],
                            logger: logging.Logger) -> Dict[Hashable, Any]:
    new_labels = {}
    non_spatial_dims = variable.dims[0:-2]
    if not non_spatial_dims:
        return new_labels

    for dim in non_spatial_dims:

        try:
            coord_var = dataset.coords[dim]
            if coord_var.size == 0:
                continue
        except KeyError:
            continue

        if labels and dim in labels:
            label = labels[str(dim)]
            try:
                label = np.array(label).astype(coord_var.dtype)
            except (TypeError, ValueError) as e:
                msg = (f'illegal label {label!r} for dimension {dim!r},'
                       f' using first label instead (error: {e})')
                if logger:
                    logger.warning(msg)
                else:
                    warnings.warn(msg)
                label = coord_var[0].values
        else:
            msg = (f'missing label for dimension {dim!r},'
                   f' using first label instead')
            if logger:
                logger.warning(msg)
            else:
                warnings.warn(msg)
            label = coord_var[0].values

        new_labels[dim] = label

    return new_labels


def _normalize_format(format: str) -> str:
    if format in ('png', 'PNG', 'image/png'):
        return 'png'
    return format


def _encode_rgba_as_png(rgba_array: np.ndarray) -> bytes:
    # noinspection PyUnresolvedReferences
    image = PIL.Image.fromarray(rgba_array)
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return bytes(stream.getvalue())
