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

from typing import Optional, List, Tuple

from xcube.constants import EARTH_EQUATORIAL_PERIMETER
from xcube.types import CoordRange

import math


def get_adjusted_res(res: float) -> float:
    if res > 180.:
        return res

    n = int(math.floor(180. / res))
    if n % 2 != 0:
        n += 1

    return 180. / n


def get_adjusted_bbox(bbox: CoordRange, res: float, size: Tuple[int, int]) -> CoordRange:
    width, height = size
    x_min, y_min, x_max, y_max = bbox
    x_min = res * round(x_min / res)
    y_min = res * round(y_min / res)
    return x_min, y_min, x_min + res * width, y_min + res * height


def compute_grid_layout(bbox: CoordRange, res: float, size: Tuple[int, int], units='degrees'):
    if res is None and bbox is None and size is None:
        raise ValueError('two of the three parameters resolution, bounding box, and size must be given')

    if units == 'degrees' or units == 'deg':
        to_degrees = identity
    elif units == 'meters' or units == 'm':
        to_degrees = meters_to_degrees
    else:
        raise ValueError('illegal units')

    if res is None:
        x_min, y_min, x_max, y_max = bbox
        width, height = size
        cov_x, cov_y = to_degrees(x_max - x_min), to_degrees(y_max - y_min)
        bbox = to_degrees(x_min), to_degrees(y_min), to_degrees(x_max), to_degrees(y_max)
        res = get_adjusted_res(min(cov_x / width, cov_y / height))
        bbox = get_adjusted_bbox(bbox, res, size)
        return dict(res=res, cov=(cov_x, cov_y), bbox=bbox, size=size)

    if bbox is None:
        res = to_degrees(res)
        res = get_adjusted_res(res)
        width, height = size
        cov_x, cov_y = res * width, res * height
        return dict(res=res, cov=(cov_x, cov_y), bbox=None, size=size)

    if size is None:
        res = to_degrees(res)
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = to_degrees(x_min), to_degrees(y_min), to_degrees(x_max), to_degrees(y_max)
        cov_x, cov_y = x_max - x_min, y_max - y_min
        size = round(cov_x / res), round(cov_y / res)
        bbox = get_adjusted_bbox(bbox, res, size)
        return dict(res=res, cov=(cov_x, cov_y), bbox=bbox, size=size)


def meters_to_degrees(res):
    return (360.0 * res) / EARTH_EQUATORIAL_PERIMETER


def degrees_to_meters(res):
    return (res / 360.0) * EARTH_EQUATORIAL_PERIMETER


def identity(x):
    return x


def main(args: Optional[List[str]] = None):
    import argparse
    import sys
    args = args or sys.argv[1:]
    parser = argparse.ArgumentParser(description='Fixed Earth Grid Calculator')
    parser.add_argument('--units', '-u', default='degrees', choices=['degrees', 'deg', 'meters', 'm'],
                        help='Coordinate units')
    parser.add_argument('--res', metavar='RES', type=float,
                        help='Desired resolution in given units')
    parser.add_argument('--bbox', metavar='BBOX',
                        help='Desired bounding box <xmin>,<ymin>,<xmax>,<ymax> in given units')
    parser.add_argument('--size', metavar='SIZE',
                        help='Desired spatial image size <width>,<height> in pixels')

    try:
        arg_obj = parser.parse_args(args)
    except SystemExit as e:
        return int(str(e))

    res = arg_obj.res
    bbox = None
    size = None
    units = arg_obj.units
    if units in ('degrees', 'deg', 'degree'):
        units = 'degrees'
    elif units in ('meters', 'meter', 'm'):
        units = 'meters'
    else:
        print('error: illegal units')
        return 2

    if arg_obj.bbox is not None:
        bbox = tuple(map(lambda s: float(s.strip()), arg_obj.bbox.split(',')))

    if arg_obj.size is not None:
        size = tuple(map(lambda s: int(s.strip()), arg_obj.size.split(',')))

    if not ((res and bbox) or (res and size) or (bbox and size)):
        print('error: two of the three parameters resolution, bounding box, and size must be given')
        return 2

    grid_layout = compute_grid_layout(bbox=bbox, res=res, size=size, units=units)
    res = grid_layout['res']
    bbox = grid_layout['bbox']
    cov = grid_layout['cov']
    size = grid_layout['size']
    print(f'Resolution in degrees:   {res}')
    print(f'Resolution in meters:    {degrees_to_meters(res)}')
    print(f'Image size in pixels:    {size[0]},{size[1]}')
    print(f'Image size in degrees:   {cov[0]},{cov[1]}')
    print(f'Image size in meters:    {degrees_to_meters(cov[0])},{degrees_to_meters(cov[1])}')
    if bbox is not None:
        print(f'Bounding box in degrees: {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}')

    return 0


if __name__ == '__main__':
    main()
