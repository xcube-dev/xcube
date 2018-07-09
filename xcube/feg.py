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

import math
from typing import Optional, List

from xcube.constants import EARTH_EQUATORIAL_RADIUS
from xcube.types import CoordRange


class FixedEarthGrid:

    def __init__(self, level_zero_res=1.0):
        level_zero_grid_size_x = int(360. / level_zero_res)
        level_zero_grid_size_y = int(180. / level_zero_res)
        msg = 'level_zero_res must divide %s by an integer number'
        if level_zero_grid_size_x * level_zero_res != 360.:
            raise ValueError(msg % 360)
        if level_zero_grid_size_y * level_zero_res != 180.:
            raise ValueError(msg % 180)
        self.level_zero_res = level_zero_res
        self.level_zero_grid_size_x = level_zero_grid_size_x
        self.level_zero_grid_size_y = level_zero_grid_size_y

    def get_res(self, level: int, units='degrees'):
        return self.from_degree(self.level_zero_res / (2 ** level), units=units)

    def get_level(self, res: float, units='degrees'):
        res = self.to_degree(res, units)
        level = int(round(math.log2(self.level_zero_res / res)))
        return level if level >= 0 else 0

    def get_level_and_res(self, res: float, units='degrees'):
        level = self.get_level(res, units=units)
        return level, self.from_degree(self.get_res(level), units=units)

    def get_grid_size(self, level: int):
        scale = 2 ** level
        return self.level_zero_grid_size_x * scale, self.level_zero_grid_size_y * scale

    def adjust_bbox(self, bbox: CoordRange, res: float, units='degrees'):
        lon_min, lat_min, lon_max, lat_max = bbox
        level, res = self.get_level_and_res(res, units=units)
        res_deg = self.to_degree(res, units)
        x_min = math.floor(lon_min / res_deg)
        y_min = math.floor(lat_min / res_deg)
        x_max = math.ceil(lon_max / res_deg)
        y_max = math.ceil(lat_max / res_deg)
        return (x_min * res_deg, y_min * res_deg, x_max * res_deg, y_max * res_deg), \
               (1 + x_max - x_min, 1 + y_max - y_min), \
               level, res

    @classmethod
    def to_degree(cls, res, units):
        if units == 'degrees' or units == 'deg':
            return res
        if units == 'meters' or units == 'm':
            return (360.0 * res) / EARTH_EQUATORIAL_RADIUS
        raise ValueError(f'unrecognized units {units!r}')

    @classmethod
    def from_degree(cls, res, units):
        if units == 'degrees' or units == 'deg':
            return res
        if units == 'meters' or units == 'm':
            return (res / 360.0) * EARTH_EQUATORIAL_RADIUS
        raise ValueError(f'unrecognized units {units!r}')


def main(args: Optional[List[str]] = None):
    import argparse
    import sys
    args = args or sys.argv[1:]
    parser = argparse.ArgumentParser(description='Fixed Earth Grid Calculator')
    parser.add_argument('--units', '-u', default='degrees', choices=['degrees', 'deg', 'meters', 'm'],
                        help='Resolution units')
    parser.add_argument('--l0res', metavar='LEVEL_ZERO_RES', default=1, type=float,
                        help='Level zero resolution in degrees')
    parser.add_argument('lon_min', type=float, help='Minimum longitude of bounding box')
    parser.add_argument('lat_min', type=float, help='Minimum latitude of bounding box')
    parser.add_argument('lon_max', type=float, help='Maximum longitude of bounding box')
    parser.add_argument('lat_max', type=float, help='Maximum latitude of bounding box')
    parser.add_argument('res', type=float, help='Output resolution in given units')

    try:
        arg_obj = parser.parse_args(args)
    except SystemExit as e:
        return int(str(e))

    lon_min = arg_obj.lon_min
    lat_min = arg_obj.lat_min
    lon_max = arg_obj.lon_max
    lat_max = arg_obj.lat_max
    res = arg_obj.res
    units = arg_obj.units

    bbox = (lon_min, lat_min, lon_max, lat_max)

    feg = FixedEarthGrid(level_zero_res=arg_obj.l0res)
    adjusted_bbox, grid_size, level, adjusted_res = feg.adjust_bbox(bbox, res, units=units)
    print(f'adjusted bbox: {adjusted_bbox}')
    print(f'adjusted res: {adjusted_res} {units}')
    print(f'grid size: {grid_size}')
    print(f'at level {level} with level zero resolution of {feg.level_zero_res} deg')
    return 0


if __name__ == '__main__':
    main()
