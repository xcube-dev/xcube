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
from typing import Optional, Tuple, Dict, Any, Hashable, Union

import PIL
import matplotlib.colors
import numpy as np
import xarray as xr

from .mldataset import MultiLevelDataset
from ..util.assertions import assert_in
from ..util.logtime import log_time
from ..util.projcache import ProjCache
from ..util.tilegrid2 import DEFAULT_TILE_SIZE
from ..util.tilegrid2 import TileGrid2
from ..util.tilegrid2 import WEB_MERCATOR_CRS_NAME

DEFAULT_VALUE_RANGE = 0, 1
DEFAULT_CMAP_NAME = 'bone'
DEFAULT_CRS_NAME = WEB_MERCATOR_CRS_NAME
DEFAULT_TILE_ENLARGEMENT = 1
DEFAULT_FORMAT = 'png'


def compute_rgba_tile(
        ml_dataset: MultiLevelDataset,
        variable_name: str,
        tile_x: int,
        tile_y: int,
        tile_z: int,
        crs_name: str = DEFAULT_CRS_NAME,
        tile_size: int = DEFAULT_TILE_SIZE,
        cmap_name: str = DEFAULT_CMAP_NAME,
        value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
        non_spatial_labels: Optional[Dict[str, Any]] = None,
        format: str = DEFAULT_FORMAT,
        tile_enlargement: int = DEFAULT_TILE_ENLARGEMENT,
        logger: Optional[logging.Logger] = None
) -> Union[bytes, np.ndarray]:
    """Compute an RGBA image tile from variable *variable_name* in
    given multi-resolution dataset *mr_dataset*.

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

    :param ml_dataset:
    :param variable_name:
    :param tile_x:
    :param tile_y:
    :param tile_z:
    :param crs_name:
    :param tile_size:
    :param cmap_name:
    :param value_range:
    :param non_spatial_labels:
    :param tile_enlargement:
    :param format:
    :param logger:
    :return: PNG bytes
    :raise TileNotFoundException
    :raise TileRequestException
    """
    assert_in(format, ('png', 'numpy'), name='format')

    with log_time(logger, 'preparing 2D subset'):

        tile_grid = TileGrid2.new(crs_name, tile_size=tile_size)

        ds_level = tile_grid.get_dataset_level(
            tile_z,
            [xy_res[1] for xy_res in ml_dataset.resolutions],
            ml_dataset.grid_mapping.spatial_unit_name
        )
        dataset = ml_dataset.get_dataset(ds_level)

        if variable_name not in dataset:
            raise TileNotFoundException.new(
                f'variable {variable_name!r}'
                f' not found in {ml_dataset.ds_id!r}',
                logger=logger
            )
        variable = dataset[variable_name]

    non_spatial_labels = _get_non_spatial_labels(dataset,
                                                 variable,
                                                 non_spatial_labels,
                                                 logger)
    if non_spatial_labels:
        variable = variable.sel(**non_spatial_labels, method='nearest')

    ds_x_name, ds_y_name = ml_dataset.grid_mapping.xy_dim_names

    ds_y_coords = variable[ds_y_name]
    ds_y_points_up = True if ds_y_coords[0] < ds_y_coords[-1] else False

    with log_time(logger,
                  'transforming tile map to dataset coordinates'):
        tile_bbox = tile_grid.get_tile_bbox(tile_x, tile_y, tile_z)
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
            tile_grid.crs,
            ml_dataset.grid_mapping.crs
        )

        tile_ds_x_2d, tile_ds_y_2d = t_map_to_ds.transform(tile_x_2d,
                                                           tile_y_2d)

    with log_time(logger, 'getting spatial subset'):

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

        dx = (tile_enlargement / tile_size) * (ds_x_max - ds_x_min)
        dy = (tile_enlargement / tile_size) * (ds_y_max - ds_y_min)
        ds_x_slice = slice(ds_x_min - dx, ds_x_max + dx)
        if ds_y_points_up:
            ds_y_slice = slice(ds_y_min - dy, ds_y_max + dy)
        else:
            ds_y_slice = slice(ds_y_max + dy, ds_y_min - dy)
        var_subset = variable.sel({ds_x_name: ds_x_slice,
                                   ds_y_name: ds_y_slice})
        # A zero or a one in the tile's shape will produce a
        # non-existing or too small image. It will also prevent
        # determining the current resolution.
        if 0 in var_subset.shape or 1 in var_subset.shape:
            return TransparentRgbaTilePool.INSTANCE.get(tile_size)

    with log_time(logger,
                  'transforming dataset coordinates into indices'):
        ds_x_coords = var_subset[ds_x_name]
        ds_y_coords = var_subset[ds_y_name]

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

    with log_time(logger, 'masking dataset indices'):
        ds_mask = (ds_x_indices >= 0) & (ds_x_indices < ds_size_x) \
                  & (ds_y_indices >= 0) & (ds_y_indices < ds_size_y)

        ds_x_indices = np.where(ds_mask, ds_x_indices, 0)
        ds_y_indices = np.where(ds_mask, ds_y_indices, 0)

    with log_time(logger, 'loading 2D data for spatial subset'):
        # Note, we need to load the values here into a numpy array,
        # because 2D indexing by [ds_y_indices, ds_x_indices]
        # does not (yet) work with dask arrays.
        var_tile = var_subset.values
        # Remove any axes above the 2nd. This is safe,
        # they will be of size one, if any.
        var_tile = var_tile.reshape(var_tile.shape[-2:])

    with log_time(logger, 'looking up dataset indices'):
        # This does the actual projection trick.
        # Lookup indices ds_y_indices, ds_x_indices to create
        # the actual tile.
        var_tile = var_tile[ds_y_indices, ds_x_indices]
        var_tile = np.where(ds_mask, var_tile, np.nan)

    with log_time(logger, 'encoding tile as RGBA image'):
        var_tile = var_tile[::-1, :]
        value_min, value_max = value_range
        if value_max < value_min:
            value_min, value_max = value_max, value_min
        if math.isclose(value_min, value_max):
            value_max = value_min + 1
        norm = matplotlib.colors.Normalize(value_min, value_max)
        cm = matplotlib.cm.get_cmap(cmap_name)
        var_tile_norm = norm(var_tile)
        var_tile_rgba = cm(var_tile_norm)
        var_tile_rgba = (255 * var_tile_rgba).astype(np.uint8)

    if format == 'png':
        with log_time(logger, 'encoding RGBA image as PNG bytes'):
            return encode_rgba_as_png(var_tile_rgba)
    else:
        return var_tile_rgba


def encode_rgba_as_png(rgba_array: np.ndarray) -> bytes:
    # noinspection PyUnresolvedReferences
    image = PIL.Image.fromarray(rgba_array)
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return bytes(stream.getvalue())


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
    """A cache for fully-transparent RGBA tiles of a given size."""

    INSTANCE: 'TransparentRgbaTilePool'

    def __init__(self):
        self._transparent_tiles: Dict[int, bytes] = dict()

    def get(self, tile_size: int) -> bytes:
        if tile_size not in self._transparent_tiles:
            data = encode_rgba_as_png(np.zeros((tile_size, tile_size, 4),
                                               dtype=np.uint8))
            self._transparent_tiles[tile_size] = data
        return self._transparent_tiles[tile_size]


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
