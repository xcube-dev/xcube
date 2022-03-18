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
from typing import Optional, Tuple, Sequence, Iterator

import pyproj

WEB_MERCATOR_CRS_NAME = 'EPSG:3857'
GEOGRAPHIC_CRS_NAME = 'EPSG:4326'
GEOGRAPHIC_CRS_NAMES = GEOGRAPHIC_CRS_NAME, 'WGS84', 'CRS84'
DEFAULT_CRS_NAME = WEB_MERCATOR_CRS_NAME
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_ENLARGEMENT = 2
EARTH_EQUATORIAL_RADIUS_WGS84 = 6378137.
EARTH_CIRCUMFERENCE_WGS84 = 2 * math.pi * EARTH_EQUATORIAL_RADIUS_WGS84


class TileGrid2:
    def __init__(self,
                 num_level_zero_tiles: Tuple[int, int],
                 crs_name: str,
                 map_height: float,
                 tile_size: int = DEFAULT_TILE_SIZE,
                 map_levels: Optional[Tuple[int]] = None,
                 map_resolutions: Optional[Tuple[float]] = None):
        self.num_level_zero_tiles = num_level_zero_tiles
        self.tile_size = tile_size
        self.crs_name = crs_name
        self.map_height = map_height
        self.map_levels = map_levels
        self.map_resolutions = map_resolutions
        self._crs = None

    @property
    def crs(self) -> pyproj.CRS:
        if self._crs is None:
            self._crs = pyproj.CRS.from_string(self.crs_name)
        return self._crs

    @property
    def map_unit_name(self) -> str:
        return self.crs.axis_info[0].unit_name

    @classmethod
    def new(cls, crs_name: str = DEFAULT_CRS_NAME, **kwargs) -> 'TileGrid2':
        if crs_name == WEB_MERCATOR_CRS_NAME:
            grid = TileGrid2.new_web_mercator()
        elif crs_name in GEOGRAPHIC_CRS_NAMES:
            grid = TileGrid2.new_geographic()
        else:
            raise ValueError(f'unsupported spatial CRS {crs_name!r}')
        return grid.derive(**kwargs)

    @classmethod
    def new_geographic(cls):
        return TileGrid2(num_level_zero_tiles=(2, 1),
                         crs_name=GEOGRAPHIC_CRS_NAME,
                         map_height=180.)

    @classmethod
    def new_web_mercator(cls):
        return TileGrid2(num_level_zero_tiles=(1, 1),
                         crs_name=WEB_MERCATOR_CRS_NAME,
                         map_height=EARTH_CIRCUMFERENCE_WGS84)

    def derive(self, **kwargs):
        args = self.to_dict()
        args.update({k: v for k, v in kwargs.items() if v is not None})
        return TileGrid2(**args)

    def to_dict(self):
        d = dict(num_level_zero_tiles=self.num_level_zero_tiles,
                 crs_name=self.crs_name,
                 map_height=self.map_height,
                 tile_size=self.tile_size,
                 map_levels=self.map_levels,
                 map_resolutions=self.map_resolutions)
        return {k: v for k, v in d.items() if v is not None}

    def resolutions(self, unit_name: Optional[str] = None) -> Iterator[float]:
        unit_factor = get_unit_factor(self.map_unit_name,
                                      unit_name or self.map_unit_name)
        res_l0 = unit_factor * self.map_height / self.tile_size
        factor = 1
        while True:
            yield res_l0 / factor
            factor *= 2

    def get_dataset_level(self,
                          level: int,
                          ds_resolutions: Sequence[float],
                          ds_resolutions_unit_name: Optional[str] = None) \
            -> int:

        map_unit_name = self.map_unit_name
        unit_factor = get_unit_factor(map_unit_name,
                                      ds_resolutions_unit_name
                                      or map_unit_name)

        map_pix_size_l0 = unit_factor * self.map_height / self.tile_size
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

    def get_tile_bbox(self,
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

        map_width = self.map_height * num_tiles_x0 / num_tiles_y0
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


def _is_meter_unit(unit_name: str) -> bool:
    return unit_name.lower() in ('m',
                                 'metre', 'metres',
                                 'meter', 'meters')


def _is_degree_unit(unit_name: str) -> bool:
    return unit_name.lower() in ('deg',
                                 'degree', 'degrees',
                                 'decimal_degree', 'decimal_degrees')
