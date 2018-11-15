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
from typing import Tuple

import click

from xcube.constants import EARTH_EQUATORIAL_PERIMETER
from xcube.version import version

LEVEL_MAX = 15
TILE_MIN = 180
INV_RES_MAX = 100


def get_adjusted_box(x1: float, y1: float, x2: float, y2: float, res: float) \
        -> Tuple[float, float, float, float]:
    adj_x1 = res * math.floor(x1 / res)
    adj_y1 = res * math.floor(y1 / res)
    adj_x2 = res * math.ceil(x2 / res)
    adj_y2 = res * math.ceil(y2 / res)
    return adj_x1, adj_y1, adj_x2, adj_y2


def meters_to_degrees(res):
    return (360.0 * res) / EARTH_EQUATORIAL_PERIMETER


def degrees_to_meters(res):
    return (res / 360.0) * EARTH_EQUATORIAL_PERIMETER


def factor_out_two(x: int) -> Tuple[int, int]:
    if x < 0:
        raise ValueError("x must not be negative")
    if x == 0:
        return 0, 0
    e = 0
    while x % 2 == 0:
        x >>= 1
        e += 1
    return x, e


def get_levels(inv_res_0: int, max_level: int) -> Tuple:
    results = []
    f = 1
    tile_size = TILE_MIN * inv_res_0
    for level in range(0, max_level + 1):
        height = f * tile_size
        inv_res = f * inv_res_0
        res = 1.0 / inv_res
        res_m = round(10 * degrees_to_meters(res)) / 10
        results.append((level, height, inv_res, res, res_m))
        f *= 2
    return tuple(results)


def find_close_resolutions(target_res: float, delta_res: float) -> Tuple:
    results = []
    seen_inv_res = set()
    for inv_res_0 in range(1, INV_RES_MAX + 1):
        f = 1
        for level in range(0, LEVEL_MAX + 1):
            inv_res = f * inv_res_0
            res = 1.0 / inv_res
            if abs(target_res - res) <= delta_res and inv_res not in seen_inv_res:
                tile_size = TILE_MIN * inv_res_0
                height = TILE_MIN * inv_res
                delta = round(1000 * (res - target_res) / target_res) / 10
                res_m = round(10 * degrees_to_meters(res)) / 10
                results.append((tile_size, level, height, inv_res, res, res_m, delta))
                seen_inv_res.add(inv_res)
            f *= 2
    return tuple(results)


@click.command(name="res")
@click.argument('target_res', metavar="TARGET_RES")
@click.option('--delta_res', '-d', metavar='DELTA_RES', default='5%',
              help='Maximum resolution delta. Defaults to "5%".')
@click.option('--sep', '-s', metavar='SEP', default='\t',
              help='Column separator for the output. Defaults to TAB.')
def list_resolutions(target_res: str, delta_res: str, sep: str):
    """
    List resolutions close to a target resolution.

    Lists possible resolutions of a fixed Earth grid that are close to a given target
    resolution TARGET_RES within a maximum allowed deviation DELTA_RES.

    Both TARGET_RES and DELTA_RES can be suffixed by a "m" to indicate meter units.
    DELTA_RES can also be suffixed by a "%" to indicate deviation from TARGET_RES in percent.

    Close resolutions are computed as RES = 1 / INV_RES = 180 / HEIGHT, where HEIGHT is the vertical number of
    grid cells of a global grid, such that HEIGHT = TILE * 2 ^ LEVEL, with LEVEL being the level number
    of a multi-resolution pyramid and TILE being the tile (or chunk) size at LEVEL zero.
    """
    if target_res.endswith("m"):
        target_res = meters_to_degrees(float(target_res[0: -1]))
    else:
        target_res = float(target_res)
    if delta_res.endswith("m"):
        delta_res = meters_to_degrees(float(delta_res[0: -1]))
    elif delta_res.endswith("%"):
        delta_res = float(delta_res[0: -1]) * target_res / 100
    else:
        delta_res = float(delta_res)

    sep = '\t' if sep.upper() == "TAB" else sep

    results = find_close_resolutions(target_res, delta_res)

    click.echo()
    click.echo(sep.join(("TILE", "LEVEL", "HEIGHT", "INV_RES", "RES (deg)", "RES (m), DELTA_RES (%)")))
    for result in results:
        click.echo(sep.join(map(str, result)))


@click.command(name="levels")
@click.argument('inv_res', metavar="INV_RES", type=int)
@click.option('--more-levels', '-m', metavar="MORE_LEVELS", type=int, default=0,
              help="Number of additional levels to list.")
@click.option('--sep', '-s', metavar='SEP', default='\t',
              help='Column separator for the output. Defaults to TAB.')
def list_levels(inv_res: int, more_levels: int, sep: str):
    """
    List levels for target resolution.

    Lists all levels and their resolutions for a given target resolution INV_RES
    which defines a fixed Earth grid.
    """
    if inv_res <= 0:
        raise click.ClickException(f"Invalid INV_RES: {inv_res}")

    sep = '\t' if sep.upper() == "TAB" else sep

    inv_res_0, target_level = factor_out_two(inv_res)
    results = get_levels(inv_res_0, target_level + more_levels)

    click.echo()
    click.echo(sep.join(("LEVEL", "HEIGHT", "INV_RES", "RES (deg)", "RES (m)")))
    for result in results:
        click.echo(sep.join(map(str, result)))


@click.command(name="abox")
@click.argument('geom', metavar="GEOM")
@click.argument('inv_res', metavar="INV_RES", type=int)
def adjust_box(geom: str, inv_res: int):
    """
    Adjust a bounding box to a fixed Earth grid.

    Adjusts a bounding box given by GEOM  to a fixed Earth grid by the
    inverse resolution INV_RES in degrees^-1 units, which must be an integer number.

    GEOM is a bounding box given as x1,y1,x2,y2 in decimal degrees.
    (Geometry WKT and GeoJSON support may be added later.)
    """
    try:
        x1, y1, x2, y2 = [float(c) for c in geom.split(",")]
    except (ValueError, TypeError) as e:
        raise click.ClickException(f"Invalid GEOM: {geom}") from e
    if inv_res <= 0:
        raise click.ClickException(f"Invalid INV_RES: {inv_res}")

    inv_res_0, level = factor_out_two(inv_res)

    tile_size = TILE_MIN * inv_res_0

    res = 1 / inv_res
    tile_cov = tile_size * res

    # Adjust along tile boundaries
    adj_x1, adj_y1, adj_x2, adj_y2 = get_adjusted_box(x1, y1, x2, y2, tile_cov)

    width = round((adj_x2 - adj_x1) / res)
    height = round((adj_y2 - adj_y1) / res)

    click.echo()
    click.echo(f'Orig. box coord. = {x1},{y1},{x2},{y2}')
    click.echo(f'Adj. box coord.  = {adj_x1},{adj_y1},{adj_x2},{adj_y2}')
    click.echo(f'Orig. box WKT    = POLYGON (('
               f'{x1} {y1},'
               f' {x2} {y1},'
               f' {x2} {y2},'
               f' {x1} {y2},'
               f' {x1} {y1}))')
    click.echo(f'Adj. box WKT     = POLYGON (('
               f'{adj_x1} {adj_y1},'
               f' {adj_x2} {adj_y1},'
               f' {adj_x2} {adj_y2},'
               f' {adj_x1} {adj_y2},'
               f' {adj_x1} {adj_y1}))')
    click.echo(f'Grid size  = {width} x {height} cells')
    click.echo('with')
    click.echo(f'  TILE      = {tile_size}')
    click.echo(f'  LEVEL     = {level}')
    click.echo(f'  INV_RES   = {inv_res}')
    click.echo(f'  RES (deg) = {res}')
    click.echo(f'  RES (m)   = {degrees_to_meters(res)}')


@click.group()
@click.version_option(version)
def cli():
    """
    The Xcube grid tool is used to find suitable spatial data cube resolutions and to
    adjust bounding boxes to that resolutions.

    We find resolutions with respect to a fixed Earth grid and adjust regional spatial
    subsets to that fixed Earth grid. We also try to select the resolutions such
    that they are taken from a certain level of a multi-resolution pyramid whose
    level resolutions increase by a factor of two.

    The graticule on the fixed Earth grid is given by

    \b
        LON(I) = -180 + I * TILE / INV_RES
        LAT(J) =  -90 + J * TILE / INV_RES

    With

    \b
        INV_RES:  An integer number greater zero.
        RES:      1 / INV_RES, the spatial grid resolution in degrees.
        TILE:     Number of grid cells of a global grid at lowest resolution level.

    Let WIDTH and HEIGHT be the number of horizontal and vertical grid cells
    of a global grid at a certain LEVEL with WIDTH * RES = 360 and HEIGHT * RES = 180, then
    we also force HEIGHT = TILE * 2 ^ LEVEL.
   """


cli.add_command(list_resolutions)
cli.add_command(list_levels)
cli.add_command(adjust_box)
