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

import math
from typing import Optional, Tuple, Sequence, List

import numpy as np
import pyproj

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance

WEB_MERCATOR_CRS_NAME = 'EPSG:3857'
GEOGRAPHIC_CRS_NAME = 'CRS84'
GEOGRAPHIC_CRS_NAMES = ('CRS84', 'EPSG:4326', 'WGS84')
DEFAULT_CRS_NAME = GEOGRAPHIC_CRS_NAME
DEFAULT_TILE_SIZE = 256
EARTH_EQUATORIAL_RADIUS_WGS84 = 6378137.
EARTH_CIRCUMFERENCE_WGS84 = 2 * math.pi * EARTH_EQUATORIAL_RADIUS_WGS84


class TileGrid:
    """
    A tile grid represents a schema for subdividing
    the world into 2D image tiles.

    :param num_level_zero_tiles: The number of tiles
        in x and y direction at lowest resolution level (zero).
    :param crs_name: Name of the spatial coordinate reference system.
    :param map_height: The height of the world map in units of the
        spatial coordinate reference system.
    :param tile_size: The tile size to be used for
        both x and y directions. Defaults to 256.
    """

    def __init__(self,
                 num_level_zero_tiles: Tuple[int, int],
                 crs_name: str,
                 map_height: float,
                 tile_size: int = DEFAULT_TILE_SIZE,
                 min_level: Optional[int] = None,
                 max_level: Optional[int] = None):
        self._num_level_zero_tiles = num_level_zero_tiles
        self._crs_name = crs_name
        self._crs = None
        self._map_height = map_height
        self._tile_size = tile_size
        self._min_level = min_level
        self._max_level = max_level

    @property
    def num_level_zero_tiles(self) -> Tuple[int, int]:
        """The number of level zero tiles in x and y directions."""
        return self._num_level_zero_tiles

    @property
    def level_zero_resolution(self) -> float:
        """The resolution at level zero in map units."""
        return self._map_height / self._tile_size

    @property
    def crs_name(self) -> str:
        """The name of the spatial coordinate reference system."""
        return self._crs_name

    @property
    def crs(self) -> pyproj.CRS:
        """The spatial coordinate reference system."""
        if self._crs is None:
            self._crs = pyproj.CRS.from_string(self.crs_name)
        return self._crs

    @property
    def map_unit_name(self) -> str:
        """The name of the map's spatial units."""
        return self.crs.axis_info[0].unit_name

    @property
    def map_width(self) -> float:
        """
        The height of the map in units
        of the map's spatial coordinate reference system.
        """
        num_tiles_x0, num_tiles_y0 = self.num_level_zero_tiles
        return self.map_height * num_tiles_x0 / num_tiles_y0

    @property
    def map_height(self) -> float:
        """
        The height of the map in units
        of the map's spatial coordinate reference system.
        """
        return self._map_height

    @property
    def map_extent(self) -> Tuple[float, float, float, float]:
        """
        The extent of the map in units
        of the map's spatial coordinate reference system.
        """
        map_width_05 = self.map_width / 2
        map_height_05 = self.map_height / 2
        return -map_width_05, -map_height_05, map_width_05, map_height_05

    @property
    def map_origin(self) -> Tuple[float, float]:
        """
        The origin of the map (upper, left pixel of the upper left tile)
        in units of the map's spatial coordinate reference system.
        """
        map_extent = self.map_extent
        return map_extent[0], map_extent[3]

    @property
    def tile_size(self) -> int:
        """The tile size in pixels."""
        return self._tile_size

    @property
    def min_level(self) -> Optional[int]:
        """The minimum level of detail."""
        return self._min_level

    @property
    def max_level(self) -> Optional[int]:
        """The maximum level of detail."""
        return self._max_level

    @property
    def num_levels(self) -> Optional[int]:
        """The number of detail levels."""
        return self.max_level + 1 if self.max_level is not None else None

    @classmethod
    def new(cls, crs_name: str = DEFAULT_CRS_NAME, **kwargs) -> 'TileGrid':
        if crs_name == WEB_MERCATOR_CRS_NAME:
            grid = TileGrid.new_web_mercator()
        elif crs_name in GEOGRAPHIC_CRS_NAMES:
            grid = TileGrid.new_geographic()
        else:
            raise ValueError(f'unsupported spatial CRS {crs_name!r}')
        return grid.derive(**kwargs)

    @classmethod
    def new_geographic(cls):
        return TileGrid(num_level_zero_tiles=(2, 1),
                        crs_name=GEOGRAPHIC_CRS_NAME,
                        map_height=180.)

    @classmethod
    def new_web_mercator(cls):
        return TileGrid(num_level_zero_tiles=(1, 1),
                        crs_name=WEB_MERCATOR_CRS_NAME,
                        map_height=EARTH_CIRCUMFERENCE_WGS84)

    def derive(self, **kwargs):
        args = self.to_dict()
        args.update({k: v for k, v in kwargs.items() if v is not None})
        return TileGrid(**args)

    def to_dict(self):
        d = dict(num_level_zero_tiles=self.num_level_zero_tiles,
                 crs_name=self.crs_name,
                 map_height=self.map_height,
                 tile_size=self.tile_size,
                 min_level=self.min_level,
                 max_level=self.max_level)
        return {k: v for k, v in d.items() if v is not None}

    def get_resolutions(self,
                        min_level: Optional[int] = None,
                        max_level: Optional[int] = None,
                        unit_name: Optional[str] = None) -> List[float]:
        unit_factor = get_unit_factor(self.map_unit_name,
                                      unit_name or self.map_unit_name)
        res_l0 = unit_factor * self.level_zero_resolution

        min_level = min_level if min_level is not None else self.min_level
        if min_level is None:
            min_level = 0
        max_level = max_level if max_level is not None else self.max_level
        if max_level is None:
            raise ValueError('max_value must be given')

        return [res_l0 / 2 ** level
                for level in range(min_level, max_level + 1)]

    def get_level_range_for_dataset(
            self,
            ds_resolutions: Sequence[float],
            ds_spatial_unit_name: str
    ) -> Tuple[int, int]:
        assert_given(ds_resolutions,
                     name='ds_resolutions')
        assert_instance(ds_spatial_unit_name, str,
                        name='ds_spatial_unit_name')

        map_res_0 = self.level_zero_resolution
        f_ds_to_map = get_unit_factor(ds_spatial_unit_name,
                                      self.map_unit_name)

        if not isinstance(ds_resolutions, np.ndarray):
            ds_resolutions = np.array(ds_resolutions)

        map_resolutions = f_ds_to_map * ds_resolutions

        map_levels = np.ceil(np.log2(map_res_0 / map_resolutions))

        min_level = np.min(map_levels)
        max_level = np.max(map_levels)

        return int(min_level), int(max_level)

    def get_dataset_level(
            self,
            level: int,
            ds_resolutions: Sequence[float],
            ds_spatial_unit_name: Optional[str] = None
    ) -> int:
        f_map_to_ds = get_unit_factor(self.map_unit_name,
                                      ds_spatial_unit_name)
        # Map tile pixel size in dataset units for tile at level 0
        map_pix_size_l0 = f_map_to_ds * self.level_zero_resolution
        # Map tile pixel size in dataset units for tile at level
        map_pix_size = map_pix_size_l0 / (1 << level)

        num_ds_levels = len(ds_resolutions)

        ds_pix_size_min = ds_resolutions[0]
        if map_pix_size <= ds_pix_size_min:
            return 0

        ds_pix_size_max = ds_resolutions[-1]
        if map_pix_size >= ds_pix_size_max:
            return num_ds_levels - 1

        for ds_level in range(num_ds_levels - 1):
            ds_pix_size_1 = ds_resolutions[ds_level]
            ds_pix_size_2 = ds_resolutions[ds_level + 1]
            if ds_pix_size_1 <= map_pix_size <= ds_pix_size_2:
                r = (map_pix_size - ds_pix_size_1) \
                    / (ds_pix_size_2 - ds_pix_size_1)
                if r < 0.5:
                    return ds_level
                else:
                    return ds_level + 1

        raise RuntimeError('should not come here')

    def get_tile_extent(self,
                        tile_x: int,
                        tile_y: int,
                        tile_z: int) \
            -> Optional[Tuple[float, float, float, float]]:
        if tile_z < 0:
            return None

        zoom_factor = 1 << tile_z

        num_tiles_x0, num_tiles_y0 = self.num_level_zero_tiles

        num_tiles_x = num_tiles_x0 * zoom_factor
        if tile_x < 0 or tile_x >= num_tiles_x:
            return None

        num_tiles_y = num_tiles_y0 * zoom_factor
        if tile_y < 0 or tile_y >= num_tiles_y:
            return None

        map_width = self.map_width
        map_height = self.map_height

        map_x0 = -map_width / 2
        map_y0 = map_height / 2

        map_tile_width = map_width / zoom_factor / num_tiles_x0
        map_tile_height = map_height / zoom_factor / num_tiles_y0

        x1 = map_x0 + tile_x * map_tile_width
        y1 = map_y0 - (tile_y + 1) * map_tile_height

        x2 = map_x0 + (tile_x + 1) * map_tile_width
        y2 = map_y0 - tile_y * map_tile_height

        return x1, y1, x2, y2


def get_unit_factor(unit_name_from: str, unit_name_to: str) -> float:
    """
    Get the factor to convert from one unit into another
    with units given by *unit_name_from* and *unit_name_to*.
    """
    from_meter = _is_meter_unit(unit_name_from)
    from_degree = _is_degree_unit(unit_name_from)
    if not from_meter and not from_degree:
        raise ValueError(f'unsupported units {unit_name_from!r}')

    to_meter = _is_meter_unit(unit_name_to)
    to_degree = _is_degree_unit(unit_name_to)
    if not to_meter and not to_degree:
        raise ValueError(f'unsupported units {unit_name_to!r}')

    if from_meter and to_degree:
        return 360 / EARTH_CIRCUMFERENCE_WGS84
    if from_degree and to_meter:
        return EARTH_CIRCUMFERENCE_WGS84 / 360
    return 1.


def subdivide_size(size: Tuple[int, int],
                   tile_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    x_size, y_size = size
    tile_size_x, tile_size_y = tile_size
    sizes = [(x_size, y_size)]
    while True:
        if x_size <= tile_size_x or y_size <= tile_size_y:
            break
        x_size = (x_size + 1) // 2
        y_size = (y_size + 1) // 2
        sizes.append((x_size, y_size))
    return sizes


def get_num_levels(size: Tuple[int, int],
                   tile_size: Tuple[int, int]) -> int:
    return len(subdivide_size(size, tile_size))


def _is_meter_unit(unit_name: str) -> bool:
    return unit_name.lower() in ('m',
                                 'metre', 'metres',
                                 'meter', 'meters')


def _is_degree_unit(unit_name: str) -> bool:
    return unit_name.lower() in ('deg',
                                 'degree', 'degrees',
                                 'decimal_degree', 'decimal_degrees')
