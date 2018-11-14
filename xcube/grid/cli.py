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

SEP = ";"
L_MAX = 15
M_MAX = 25
T_MIN = 180
T_MAX = M_MAX * T_MIN


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


def subdivide_inv_res(inv_res: int, target_level: int = None) -> Tuple[int, int]:
    height = inv_res * T_MIN
    tile_size = height
    level = 0
    while tile_size % 2 == 0 and (tile_size // 2) % T_MIN == 0:
        tile_size //= 2
        level += 1
        if level == target_level:
            break
    return tile_size, level


def list_resolutions(tile_size: int, max_level: int) -> Tuple:
    results = []
    f = 1
    for level in range(0, max_level + 1):
        height = f * tile_size
        inv_res = height // T_MIN
        res = 1.0 / inv_res
        results.append((level, height, inv_res, res, degrees_to_meters(res)))
        f *= 2
    return tuple(results)


def find_close_resolutions(target_res: float, delta_res: float) -> Tuple:
    results = []
    seen_inv_res = set()
    for tile_size in range(T_MIN, T_MAX + 1):
        f = 1
        for level in range(0, L_MAX + 1):
            height = tile_size * f
            if height % T_MIN == 0:
                inv_res = height // T_MIN
                if inv_res not in seen_inv_res:
                    res = 1.0 / inv_res
                    if abs(target_res - res) <= delta_res:
                        results.append((tile_size, level, height, inv_res, res, degrees_to_meters(res)))
                        seen_inv_res.add(inv_res)
            f *= 2
    return tuple(results)


@click.command(name="fres")
@click.argument('target_res', metavar="TARGET_RES")
@click.option('--delta_res', '-d', metavar='DELTA_RES', default='10m',
              help='Maximum resolution delta. Defaults to "10m".')
@click.option('--sep', '-s', metavar='SEP', default='\t',
              help='Column separator for the output. Defaults to TAB.')
def find_resolutions(target_res: str, delta_res: str, sep: str):
    """
    Find close resolutions to a given target resolution TARGET_RES within a DELTA_RES.

    Both TARGET_RES and DELTA_RES can be suffixed by a "m" to indicate meter units.

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
    else:
        delta_res = float(delta_res)

    sep = '\t' if sep.upper() == "TAB" else sep

    results = find_close_resolutions(target_res, delta_res)

    click.echo()
    click.echo(sep.join(("TILE", "LEVEL", "HEIGHT", "INV_RES", "RES (deg)", "RES (m)")))
    for result in results:
        click.echo(sep.join(map(str, result)))


@click.command(name="lres")
@click.argument('inv_res', metavar="INV_RES", type=int)
@click.option('--max-level', '-m', metavar="MAX_LEVEL", type=int, help="Maximum level to list.")
@click.option('--sep', '-s', metavar='SEP', default='\t',
              help='Column separator for the output. Defaults to TAB.')
def list_resolutions(inv_res: int, max_level: int, sep: str):
    """
    List up to MAX_LEVEL resolutions for a given target resolution INV_RES.

    Close resolutions are computed as RES = 1 / INV_RES = 180 / HEIGHT, where HEIGHT is the vertical number of
    grid cells of a global grid, such that HEIGHT = TILE * 2 ^ LEVEL, with LEVEL being the level number
    of a multi-resolution pyramid and TILE being the tile (or chunk) size at LEVEL zero.
    """
    if inv_res <= 0:
        raise click.ClickException(f"Invalid INV_RES: {inv_res}")

    target_inv_res = inv_res
    target_res = 1.0 / target_inv_res
    sep = '\t' if sep.upper() == "TAB" else sep

    tile_size, target_level = subdivide_inv_res(inv_res, target_level=max_level)
    max_level = max(max_level, target_level) if max_level is not None else target_level
    results = list_resolutions(tile_size, max_level)

    click.echo()
    click.echo(sep.join(("LEVEL", "HEIGHT", "INV_RES", "RES (deg)", "RES (m)")))
    for result in results:
        click.echo(sep.join(map(str, result)))
    click.echo()
    click.echo("Actual:")
    click.echo(f"  TILE    = {tile_size}")
    click.echo(f"  LEVEL   = {target_level}")
    click.echo(f"  INV_RES = {target_inv_res}")
    click.echo(f"  RES     = {target_res}")
    click.echo(f"  RES (m) = {degrees_to_meters(target_res)}")


@click.command(name="abox")
@click.argument('geom', metavar="GEOM")
@click.argument('inv_res', metavar="INV_RES", type=int)
@click.option('--level', '-l', metavar="LEVEL", type=int, help="Target level.")
def adjust_box(geom: str, inv_res: int, level: int):
    """
    Adjust bounding box of the given geometry GEOM for a given
    inverse resolution INV_RES in 1/degree units which must be an integer number.

    GEOM is a bounding box given as x1,y1,x2,y2 in decimal degrees.
    (Geometry WKT and GeoJSON support may be added later.)

    The new coordinates are adjusted on a graticule given by

        LON(I) = -180 + I * RES * TILE, LAT(J) = -90 + J * RES * TILE.

    The global grid size is given by WIDTH and HEIGHT as

        WIDTH = 2 * HEIGHT, HEIGHT = 180 * INV_RES = TILE * 2 ^ LEVEL.

    """
    try:
        x1, y1, x2, y2 = [float(c) for c in geom.split(",")]
    except (ValueError, TypeError) as e:
        raise click.ClickException(f"Invalid GEOM: {geom}") from e
    if inv_res <= 0:
        raise click.ClickException(f"Invalid INV_RES: {inv_res}")

    tile_size, level = subdivide_inv_res(inv_res, target_level=level)

    res = 1 / inv_res
    tile_cov = tile_size * res

    # Adjust along tile boundaries
    adj_x1, adj_y1, adj_x2, adj_y2 = get_adjusted_box(x1, y1, x2, y2, tile_cov)

    width = round((adj_x2 - adj_x1) / res)
    height = round((adj_y2 - adj_y1) / res)

    click.echo(f'Box coord. = {adj_x1},{adj_y1},{adj_x2},{adj_y2}')
    click.echo(f'Box WKT    = POLYGON (('
               f'{adj_x1} {adj_y1},'
               f' {adj_x2} {adj_y1},'
               f' {adj_x2} {adj_y2},'
               f' {adj_x1} {adj_y2},'
               f' {adj_x1} {adj_y1}))')
    click.echo(f'Grid size  = {width} x {height} cells')
    click.echo('with')
    click.echo(f'  TILE      = {tile_size}')
    click.echo(f'  LEVEL     = {level}')
    click.echo(f'  RES (deg) = {res}')
    click.echo(f'  RES (m)   = {degrees_to_meters(res)}')


@click.group()
@click.version_option(version)
def cli():
    """
    Grid tool.
    """


cli.add_command(find_resolutions)
cli.add_command(list_resolutions)
cli.add_command(adjust_box)
