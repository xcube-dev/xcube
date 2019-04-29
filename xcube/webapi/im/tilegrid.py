# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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
import math
from typing import Optional, Any, Tuple

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

MODE_LE = -1
MODE_EQ = 0
MODE_GE = 1

GeoExtent = Tuple[float, float, float, float]

GLOBAL_GEO_EXTENT = -180.0, -90.0, +180.0, +90.0


class TileGrid:
    """
    Image pyramid tile grid.

    :param num_levels: Number of pyramid levels.
    :param num_level_zero_tiles_x: Number of tiles at level zero in X direction.
    :param num_level_zero_tiles_y:  Number of tiles at level zero in Y direction.
    :param tile_width: The tile width.
    :param tile_height: The tile height.
    :param geo_extent: The geographical extent.
    """

    def __init__(self,
                 num_levels: int,
                 num_level_zero_tiles_x: int,
                 num_level_zero_tiles_y: int,
                 tile_width: int,
                 tile_height: int,
                 geo_extent: GeoExtent,
                 inv_y: bool = False):
        if num_levels < 1:
            raise ValueError(f"{num_levels} is an invalid value for num_levels")
        if num_level_zero_tiles_x < 1:
            raise ValueError(f"{num_level_zero_tiles_x} is an invalid value for num_level_zero_tiles_x")
        if num_level_zero_tiles_y < 1:
            raise ValueError(f"{num_level_zero_tiles_y} is an invalid value for num_level_zero_tiles_x")
        if tile_width < 1:
            raise ValueError(f"{tile_width} is an invalid value for tile_width")
        if tile_height < 1:
            raise ValueError(f"{tile_height} is an invalid value for tile_height")
        west, south, east, north = geo_extent
        if west < -180.0 or south < -90.0 or east > 180.0 or north > 90.0 \
                or west == east or south >= north:
            raise ValueError(f"{geo_extent} is an invalid value for geo_extent")
        self.num_levels = num_levels
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.num_level_zero_tiles_x = num_level_zero_tiles_x
        self.num_level_zero_tiles_y = num_level_zero_tiles_y
        self.geo_extent = west, south, east, north
        self.inv_y = inv_y

    def num_tiles(self, level: int) -> Tuple[int, int]:
        return self.num_tiles_x(level), self.num_tiles_y(level)

    def num_tiles_x(self, level: int) -> int:
        return self.num_level_zero_tiles_x * (1 << level)

    def num_tiles_y(self, level: int) -> int:
        return self.num_level_zero_tiles_y * (1 << level)

    def size(self, level: int) -> Tuple[int, int]:
        return self.width(level), self.height(level)

    def width(self, level: int) -> int:
        return self.num_tiles_x(level) * self.tile_width

    def height(self, level: int) -> int:
        return self.num_tiles_y(level) * self.tile_height

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self.tile_width, self.tile_height

    @property
    def min_width(self) -> int:
        return self.width(0)

    @property
    def min_height(self) -> int:
        return self.height(0)

    @property
    def max_width(self) -> int:
        return self.width(self.num_levels - 1)

    @property
    def max_height(self) -> int:
        return self.height(self.num_levels - 1)

    def __hash__(self) -> int:
        return self.num_levels \
               + 2 * self.tile_width \
               + 4 * self.tile_height \
               + 8 * self.num_level_zero_tiles_x \
               + 16 * self.num_level_zero_tiles_y \
               + 32 * int(self.inv_y) \
               + hash(self.geo_extent)  # noqa: E126

    def __eq__(self, o: Any) -> bool:
        try:
            return self.num_levels == o.num_levels \
                   and self.tile_width == o.tile_width \
                   and self.tile_height == o.tile_height \
                   and self.num_level_zero_tiles_x == o.num_level_zero_tiles_x \
                   and self.num_level_zero_tiles_y == o.num_level_zero_tiles_y \
                   and self.inv_y == o.inv_y \
                   and self.geo_extent == o.geo_extent  # noqa: E126
        except AttributeError:
            return False

    def __str__(self):
        return '\n'.join([f'number of pyramid levels: {self.num_levels}',
                          f'number of tiles at level zero: {self.num_level_zero_tiles_x} x {self.num_level_zero_tiles_y}',
                          f'pyramid tile size: {self.tile_width} x {self.tile_height}',
                          f'image size at level zero: {self.min_width} x {self.min_height}',
                          f'image size at level {self.num_levels - 1}: {self.max_width} x {self.max_height}',
                          f'geographic extent: {self.geo_extent}',
                          f'y-axis points down: {"no" if self.inv_y else "yes"}'])

    def __repr__(self):
        return (f'TileGrid({self.num_levels}, '
                f'{self.num_level_zero_tiles_x}, '
                f'{self.num_level_zero_tiles_y}, '
                f'{self.tile_width}, '
                f'{self.tile_height}, '
                f'{repr(self.geo_extent)}, '
                f'inv_y={self.inv_y})')

    def to_json(self):
        return dict(numLevelZeroTilesX=self.num_level_zero_tiles_x,
                    numLevelZeroTilesY=self.num_level_zero_tiles_y,
                    tileWidth=self.tile_width,
                    tileHeight=self.tile_height,
                    numLevels=self.num_levels,
                    invY=self.inv_y,
                    extent=dict(west=self.geo_extent[0],
                                south=self.geo_extent[1],
                                east=self.geo_extent[2],
                                north=self.geo_extent[3]))

    @classmethod
    def create(cls,
               w: int, h: int,
               tile_width: Optional[int], tile_height: Optional[int],
               geo_extent: GeoExtent,
               inv_y: bool = False) -> 'TileGrid':
        """
        Create a new TileGrid.

        :param w: original image width
        :param h: original image height
        :param tile_width: optimal tile width
        :param tile_height: optimal tile height
        :param geo_extent: The geo-spatial extent
        :param inv_y: True, if the positive y-axis (latitude) points up
        :return: A new TilingScheme object
        """
        west, south, east, north = map(_adjust_to_floor, geo_extent)
        w_mode = MODE_GE
        if west == -180.0 and east == 180.0:
            w_mode = MODE_EQ
        h_mode = MODE_GE
        if south == -90.0 and north == 90.0:
            h_mode = MODE_EQ

        (w_new, h_new), (tw, th), (nt0x, nt0y), nl = pow2_2d_subdivision(w, h,
                                                                         w_mode=w_mode,
                                                                         h_mode=h_mode,
                                                                         tw_opt=min(w, tile_width or 256),
                                                                         th_opt=min(h, tile_height or 256))

        new_extent = cls._adjust_geo_extent((west, south, east, north), w, h, w_new, h_new, inv_y=inv_y)
        return TileGrid(nl, nt0x, nt0y, tw, th, new_extent, inv_y=inv_y)

    @classmethod
    def _adjust_geo_extent(cls, geo_extent: GeoExtent, w_old, h_old, w_new, h_new, inv_y: bool = False) -> GeoExtent:

        assert w_new >= w_old
        assert h_new >= h_old

        lon1, lat1, lon2, lat2 = geo_extent

        if lon1 < lon2:
            # not crossing anti-meridian
            delta_lon = lon2 - lon1
        else:
            # crossing anti-meridian
            delta_lon = 360. + lon2 - lon1
        delta_lat = lat2 - lat1

        if w_new > w_old:
            delta_lon_new = w_new * delta_lon / w_old
            # We cannot adjust lon1, because we expect x to increase with x indices
            # and hence we would later on have to read from negative x indexes
            lon2_new = lon1 + delta_lon_new
            if lon2_new > 180.:
                lon2_new = lon2_new - 360.
        else:
            lon2_new = lon2

        if h_new > h_old:
            delta_lat_new = h_new * delta_lat / h_old
            if inv_y:
                # We cannot adjust lat2, because we expect y to decrease with y indices
                # and hence we would later on have to read from negative y indexes
                lat2_new = lat2
                lat1_new = lat2_new - delta_lat_new
            else:
                # We cannot adjust lat1, because we expect y to increase with y indices
                # and hence we would later on have to read from negative y indexes
                lat1_new = lat1
                lat2_new = lat1_new + delta_lat_new
        else:
            lat1_new, lat2_new = lat1, lat2

        return lon1, lat1_new, lon2_new, lat2_new


def _adjust_to_floor(x: float) -> float:
    fx = math.floor(x)
    return fx if abs(fx - x) < 1e-10 else x


@functools.lru_cache(maxsize=256)
def pow2_2d_subdivision(w: int, h: int,
                        w_mode: int = MODE_EQ, h_mode: int = MODE_EQ,
                        tw_opt: Optional[int] = None, th_opt: Optional[int] = None,
                        tw_min: Optional[int] = None, th_min: Optional[int] = None,
                        tw_max: Optional[int] = None, th_max: Optional[int] = None,
                        nt0_max: Optional[int] = None,
                        nl_max: Optional[int] = None):
    """
    Get a pyramidal quad-tree subdivision of a 2D image rectangle given by image width *w* and height *h*.
    We want all pyramid levels to use the same tile size *tw*, *th*. All but the lowest resolution level, level zero,
    shall have 2 times the number of tiles of a previous level in both x- and y-direction.

    As there can be multiple of such subdivisions, we select an optimum subdivision by constraints. We want
    (in this order):
    1. the resolution of the highest pyramid level, *nl* - 1, to be as close as possible to *w*, *h*;
    2. the number of tiles in level zero to be as small as possible;
    3. the tile sizes *tw*, *th* to be as close as possible to *tw_opt*, *th_opt*, if given;
    4. a maximum number of levels.

    :param w: image width
    :param h: image height
    :param w_mode: optional mode for horizontal direction, -1: *w_act* <= *w*, 0: *w_act* == *w*, +1: *w_act* >= *w*
    :param h_mode: optional mode for vertical direction, -1: *h_act* <= *h*, 0: *h_act* == *h*, +1: *h_act* >= *h*
    :param tw_opt: optional optimum tile width
    :param th_opt: optional optimum tile height
    :param tw_min: optional minimum tile width
    :param th_min: optional minimum tile height
    :param tw_max: optional maximum tile width
    :param th_max: optional maximum tile height
    :param nt0_max: optional maximum number of tiles at level zero of pyramid
    :param nl_max: optional maximum number of pyramid levels
    :return: a tuple ((*w_act*, *h_act*), (*tw*, *th*), (*nt0_x*, *nt0_y*), *nl*) with
             *w_act*, *h_act* being the final image width and height in the pyramids's highest resolution level;
             *tw*, *th* being the tile width and height;
             *nt0_x*, *nt0_y* being the number of tiles at level zero of pyramid in horizontal and vertical direction;
             and *nl* being the total number of pyramid levels.
    """
    w_act, tw, nt0_x, nl_x = pow2_1d_subdivision(w, s_mode=w_mode,
                                                 ts_opt=tw_opt, ts_min=tw_min, ts_max=tw_max,
                                                 nt0_max=nt0_max, nl_max=nl_max)
    h_act, th, nt0_y, nl_y = pow2_1d_subdivision(h, s_mode=h_mode,
                                                 ts_opt=th_opt, ts_min=th_min, ts_max=th_max,
                                                 nt0_max=nt0_max, nl_max=nl_max)
    if nl_x < nl_y:
        nl = nl_x
        f = 1 << (nl - 1)
        h0 = (h_act + f - 1) // f
        nt0_y = (h0 + th - 1) // th
    elif nl_x > nl_y:
        nl = nl_y
        f = 1 << (nl - 1)
        w0 = (w_act + f - 1) // f
        nt0_x = (w0 + tw - 1) // tw
    else:
        nl = nl_x

    return (w_act, h_act), (tw, th), (nt0_x, nt0_y), nl


def pow2_1d_subdivision(s_act: int,
                        s_mode: int = MODE_EQ,
                        ts_opt: Optional[int] = None,
                        ts_min: Optional[int] = None,
                        ts_max: Optional[int] = None,
                        nt0_max: Optional[int] = None,
                        nl_max: Optional[int] = None):
    return pow2_1d_subdivisions(s_act,
                                s_mode=s_mode,
                                ts_opt=ts_opt,
                                ts_min=ts_min, ts_max=ts_max,
                                nt0_max=nt0_max, nl_max=nl_max)[0]


def pow2_1d_subdivisions(s: int,
                         s_mode: int = MODE_EQ,
                         ts_opt: Optional[int] = None,
                         ts_min: Optional[int] = None,
                         ts_max: Optional[int] = None,
                         nt0_max: Optional[int] = None,
                         nl_max: Optional[int] = None):
    if s is None or s < 1:
        raise ValueError('invalid s')

    if s == ts_opt:
        return [(s, s, 1, 1)]

    ts_min = ts_min or min(s, (ts_opt // 2 if ts_opt else 200))
    ts_max = ts_max or min(s, (ts_opt * 2 if ts_opt else 1200))
    nt0_max = nt0_max or 8
    nl_max = nl_max or 16

    if ts_min < 1:
        raise ValueError('invalid ts_min')
    if ts_max < 1:
        raise ValueError('invalid ts_max')
    if ts_opt is not None and ts_opt < 1:
        raise ValueError('invalid ts_opt')
    if nt0_max < 1:
        raise ValueError('invalid nt0_max')
    if nl_max < 1:
        raise ValueError('invalid nl_max')

    subdivisions = []
    for ts in range(ts_min, ts_max + 1):
        s_max_min = s if s_mode == MODE_EQ or s_mode == MODE_GE else s - (ts - 1)
        s_max_max = s if s_mode == MODE_EQ or s_mode == MODE_LE else s + (ts - 1)
        for nt0 in range(1, nt0_max):
            s_max = nt0 * ts
            if s_max > s_max_max:
                break
            for nl in range(2, nl_max):
                nt = (1 << (nl - 1)) * nt0
                s_max = nt * ts
                ok = False
                if s_mode == MODE_GE:
                    if s_max >= s:
                        if s_max > s_max_max:
                            break
                        ok = True
                elif s_mode == MODE_LE:
                    if s >= s_max >= s_max_min:
                        ok = True
                else:  # s_mode == MODE_EQ:
                    if s_max == s:
                        ok = True
                    elif s_max > s:
                        break
                if ok:
                    rec = s_max, ts, nt0, nl
                    subdivisions.append(rec)

    if not subdivisions:
        return [(s, s, 1, 1)]

    # maximize nl
    subdivisions.sort(key=lambda r: r[3], reverse=True)
    if ts_opt:
        # minimize |ts - ts_opt|
        subdivisions.sort(key=lambda r: abs(r[1] - ts_opt))
    # minimize nt0
    subdivisions.sort(key=lambda r: r[2])
    # minimize s_max - s_min
    subdivisions.sort(key=lambda r: r[0] - s)

    return subdivisions
