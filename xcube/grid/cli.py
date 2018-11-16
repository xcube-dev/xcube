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

import fractions
import math
from typing import Tuple, List, Optional

import click

from xcube.constants import EARTH_EQUATORIAL_PERIMETER
from xcube.version import version

_DEFAULT_MIN_LEVEL = 0
_DEFAULT_MAX_TILE = 2500
_DEFAULT_RES_DELTA = "2.5%"
_DEFAULT_SORT_BY = "R_D"
_DEFAULT_LAT_COVERAGE = fractions.Fraction(180, 1)

_SORT_BY_KEYS_0 = ['R_D', 'R_NOM', 'R_DEN', 'R', 'H', 'H0', 'L']
_SORT_BY_KEYS_P = ["+" + k for k in _SORT_BY_KEYS_0]
_SORT_BY_KEYS_M = ["-" + k for k in _SORT_BY_KEYS_0]
_SORT_BY_KEYS = _SORT_BY_KEYS_0 + _SORT_BY_KEYS_P + _SORT_BY_KEYS_M


def find_close_resolutions(target_res: float,
                           delta_res: float,
                           coverage: fractions.Fraction,
                           max_height_0: int = _DEFAULT_MAX_TILE,
                           min_level: int = _DEFAULT_MIN_LEVEL,
                           int_inv_res: bool = False,
                           sort_by: str = _DEFAULT_SORT_BY) -> List[Tuple]:
    if target_res <= 0.0:
        raise ValueError('illegal target_res')
    if delta_res < 0.0 or delta_res >= target_res:
        raise ValueError('illegal delta_res')
    if min_level < 0.0:
        raise ValueError('illegal min_level')
    header = ("R_D(%)", "R_NOM", "R_DEN", "R(degrees)", "R(m)", "H", "H0", "L")
    reverse_sort = False
    if sort_by.startswith("+") or sort_by.startswith("-"):
        reverse_sort = sort_by[0] == "-"
        sort_by = sort_by[1:]
    if sort_by == "R_D":
        def sort_key(item):
            return abs(item[0])
    elif sort_by == "R_NOM":
        def sort_key(item):
            return item[1]
    elif sort_by == "R_DEN":
        def sort_key(item):
            return item[2]
    elif sort_by == "R":
        def sort_key(item):
            return item[3]
    elif sort_by == "H":
        def sort_key(item):
            return item[5]
    elif sort_by == "H0":
        def sort_key(item):
            return item[6]
    elif sort_by == "L":
        def sort_key(item):
            return item[7]
    else:
        raise ValueError(f'illegal sort key: {sort_by}')
    # Compute h_1, h_2, the range of possible integer heights
    res_1 = target_res - delta_res
    res_2 = target_res + delta_res
    h_1 = coverage / res_1
    h_2 = coverage / res_2
    if h_2 < h_1:
        h_1, h_2 = h_2, h_1
    h_1 = int(math.floor(h_1))
    h_2 = int(math.ceil(h_2))
    # Collect resolutions all possible integer heights
    data = []
    for height in range(h_1, h_2 + 1):
        res = coverage / fractions.Fraction(height, 1)
        # We may only want resolutions whose inverse is integer, e.g. 1/12 degree
        if not int_inv_res or res.numerator == 1:
            res_f = float(res)
            delta = res_f - target_res
            # Only if we are within delta_res
            if abs(delta) <= delta_res:
                # Only if res * h = coverage
                if res * height == coverage:
                    height_0, level = factor_out_two(height)
                    # Only if tile size falls below max and level exceeds min
                    if height_0 <= max_height_0 and level >= min_level:
                        delta_p = _round(100 * delta / target_res, 1000)
                        res_m = _round(degrees_to_meters(res_f), 10)
                        data.append((delta_p, res.numerator, res.denominator, res_f, res_m, height, height_0, level))
    data = sorted(data, key=sort_key, reverse=reverse_sort)
    return [header] + data


def get_levels(height: int, coverage: fractions.Fraction, level_min: int) -> List[Tuple]:
    res = coverage / fractions.Fraction(height)
    height_0, level = factor_out_two(height)
    data = []
    f = 1
    res_0 = res * (2 ** level)
    for i in range(0, max(level, level_min or level) + 1):
        height_i = height_0 * f
        res_i = res_0 / f
        data.append((i, height_i, res_i.numerator, res_i.denominator, float(res_i)))
        f *= 2
    header = ("L", "H", "R_NUM", "R_DEN", "R")
    return [header] + data


def get_adjusted_box(x1: float, y1: float, x2: float, y2: float, res: float) \
        -> Tuple[float, float, float, float]:
    adj_x1 = res * math.floor(x1 / res)
    adj_y1 = res * math.floor(y1 / res)
    adj_x2 = res * math.ceil(x2 / res)
    adj_y2 = res * math.ceil(y2 / res)
    if adj_x2 - res >= x2:
        adj_x2 -= res
    if adj_y2 - res >= y2:
        adj_y2 -= res
    return adj_x1, adj_y1, adj_x2, adj_y2


def meters_to_degrees(res):
    return (360.0 * res) / EARTH_EQUATORIAL_PERIMETER


def degrees_to_meters(res):
    return (res / 360.0) * EARTH_EQUATORIAL_PERIMETER


def _round(x: float, n: int) -> float:
    return round(n * x) / n


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


@click.command(name="res")
@click.argument('target_res', metavar="TARGET_RES")
@click.option('--delta_res', '-d', metavar='DELTA_RES', default=_DEFAULT_RES_DELTA,
              help=f'Maximum resolution delta. Defaults to {_DEFAULT_RES_DELTA}.')
@click.option('--coverage', '-c', metavar="COVERAGE", default=str(_DEFAULT_LAT_COVERAGE),
              help=f'The vertical coverage in degrees. Defaults to {_DEFAULT_LAT_COVERAGE} degrees.')
@click.option('--tile_max', '-t', metavar='TILE_MAX', default=_DEFAULT_MAX_TILE, type=int,
              help=f'Maximum tile size. Defaults to {_DEFAULT_MAX_TILE}.')
@click.option('--level_min', '-l', metavar='LEVEL_MIN', default=_DEFAULT_MIN_LEVEL, type=int,
              help=f'Minimum resolution level. Defaults to {_DEFAULT_MIN_LEVEL}.')
@click.option('--int_inv_res', '-i', metavar='INT_INV_RES', is_flag=True,
              help=f'Find only resolutions whose inverse are integers.')
@click.option('--sort_by', '-s', metavar='SORT_BY',
              type=click.Choice(_SORT_BY_KEYS), default=_DEFAULT_SORT_BY,
              help='Sort output by column name.')
@click.option('--sep', metavar='SEP', default='\t',
              help='Column separator for the output. Defaults to TAB.')
def list_resolutions(target_res: str,
                     delta_res: str,
                     coverage: str,
                     tile_max: int,
                     level_min: int,
                     int_inv_res: bool,
                     sort_by: str,
                     sep: str):
    """
    List resolutions close to target resolution.

    Lists possible resolutions of a fixed Earth grid that are close to a given target
    resolution TARGET_RES within a maximum allowed deviation DELTA_RES.

    Both TARGET_RES and DELTA_RES can be suffixed by a "m" to indicate meter units.
    DELTA_RES can also be suffixed by a "%" to indicate deviation from TARGET_RES in percent.

    If LEVEL_MIN is provided and greater zero, only resolutions are listed whose
    HEIGHT is larger than TILE * 2 ^ LEVEL_MIN.
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

    coverage = _fetch_coverage_from_option(coverage)

    sep = '\t' if sep.upper() == "TAB" else sep

    results = find_close_resolutions(target_res,
                                     delta_res,
                                     coverage,
                                     max_height_0=tile_max,
                                     min_level=level_min,
                                     int_inv_res=int_inv_res,
                                     sort_by=sort_by)

    click.echo()
    for result in results:
        click.echo(sep.join(map(str, result)))


@click.command(name="levels")
@click.option('--res', '-r', metavar="RES",
              help='Resolution in degrees. Can also be a rational number of form RES_NOM/RES_DEN.')
@click.option('--height', '-h', metavar="HEIGHT", type=int,
              help='Height in grid cells.')
@click.option('--coverage', '-c', metavar="COVERAGE", default=str(_DEFAULT_LAT_COVERAGE),
              help=f'The vertical coverage in degrees. Defaults to {_DEFAULT_LAT_COVERAGE} degrees.')
@click.option('--level_min', '-l', metavar="LEVEL_MIN", type=int,
              help="List at least up to this level.")
@click.option('--sep', metavar='SEP', default='\t',
              help='Column separator for the output. Defaults to TAB.')
def list_levels(res: str, height: int, coverage: str, level_min: Optional[int], sep: str):
    """
    List levels for resolution or tile size.

    Lists the given number of LEVELS for given resolution RES or given height in grid cells HEIGHT.
    which can both be used to define a fixed Earth grid.
    """
    height, coverage = _fetch_height_and_coverage_from_options(res, height, coverage)

    sep = '\t' if sep.upper() == "TAB" else sep

    rows = get_levels(height, coverage, level_min)

    click.echo()
    for row in rows:
        click.echo(sep.join(map(str, row)))


@click.command(name="abox")
@click.argument('geom', metavar="GEOM")
@click.option('--res', '-r', metavar="RES",
              help='Resolution in degrees. Can also be a rational number of form RES_NOM/RES_DEN.')
@click.option('--height', '-h', metavar="HEIGHT", type=int,
              help='Height in grid cells.')
@click.option('--coverage', '-c', metavar="COVERAGE", default=str(_DEFAULT_LAT_COVERAGE),
              help=f'The vertical coverage in degrees. Defaults to {_DEFAULT_LAT_COVERAGE} degrees.')
def adjust_box(geom: str, res: Optional[str], height: Optional[int], coverage: str):
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

    height, coverage = _fetch_height_and_coverage_from_options(res, height, coverage)

    height_0, level = factor_out_two(height)

    # coverage is height_0 in degrees
    res = coverage / fractions.Fraction(height, 1)
    coverage = height_0 * float(res)

    # Adjust along tile boundaries
    adj_x1, adj_y1, adj_x2, adj_y2 = get_adjusted_box(x1, y1, x2, y2, coverage)

    reg_width = round((adj_x2 - adj_x1) / float(res))
    reg_height = round((adj_y2 - adj_y1) / float(res))

    orig_coords = (f'(({x1} {y1},'
                   f' {x2} {y1},'
                   f' {x2} {y2},'
                   f' {x1} {y2},'
                   f' {x1} {y1}))')
    adj_coords = (f'(({adj_x1} {adj_y1},'
                  f' {adj_x2} {adj_y1},'
                  f' {adj_x2} {adj_y2},'
                  f' {adj_x1} {adj_y2},'
                  f' {adj_x1} {adj_y1}))')

    click.echo()
    click.echo(f'Orig. box coord. = {x1},{y1},{x2},{y2}')
    click.echo(f'Adj. box coord.  = {adj_x1},{adj_y1},{adj_x2},{adj_y2}')
    click.echo(f'Orig. box WKT    = POLYGON {orig_coords}')
    click.echo(f'Adj. box WKT     = POLYGON {adj_coords}')
    click.echo(f'Combined WKT     = MULTIPOLYGON ({orig_coords}, {adj_coords})')
    click.echo(f'Box grid size    = {reg_width} x {reg_height} cells')
    click.echo('Grid defined by')
    click.echo(f'  H0      = {height_0}')
    click.echo(f'  L       = {level}')
    click.echo(f'  R (deg) = {res}')
    click.echo(f'  R (m)   = {_round(degrees_to_meters(res), 100)}')


def _fetch_height_and_coverage_from_options(res_str: Optional[str],
                                            height: Optional[int],
                                            coverage_str: str) -> Tuple[int, fractions.Fraction]:
    coverage = _fetch_coverage_from_option(coverage_str)
    if res_str is not None:
        if height is not None:
            raise click.ClickException(f"Either RES or HEIGHT must be given, not both")
        try:
            res = fractions.Fraction(res_str)
        except ValueError as e:
            raise click.ClickException(f"Invalid RES: {res_str}") from e
        if res <= 0:
            raise click.ClickException(f"Invalid RES: {res_str}")
        height = coverage / res
        if height.denominator != 1:
            raise click.ClickException(f"Invalid RES: {res_str}, {coverage_str}/RES must be an integer number.")
        height = height.numerator
    elif height is None:
        raise click.ClickException(f"Either RES or HEIGHT must be given.")
    return height, coverage


def _fetch_coverage_from_option(coverage_str: str) -> fractions.Fraction:
    try:
        coverage = fractions.Fraction(coverage_str)
    except ValueError as e:
        raise click.ClickException(f"Invalid COVERAGE: {coverage_str}") from e
    if coverage <= 0:
        raise click.ClickException(f"Invalid COVERAGE: {coverage_str}")

    return coverage


@click.group()
@click.version_option(version)
def cli():
    """
    The Xcube grid tool is used to find suitable spatial data cube resolutions and to
    adjust bounding boxes to that resolutions.

    We find resolutions with respect to a possibly regional fixed Earth grid and adjust regional spatial
    subsets to that grid. We also try to select the resolutions such
    that they are taken from a certain level of a multi-resolution pyramid whose
    level resolutions increase by a factor of two.

    The graticule at a given resolution level L within the grid is given by

    \b
        RES(L) = COVERAGE * HEIGHT(L)
        HEIGHT(L) = HEIGHT_0 * 2 ^ L
        LON(L, I) = LON_MIN + I * HEIGHT_0 * RES(L)
        LAT(L, J) = LAT_MIN + J * HEIGHT_0 * RES(L)

    With

    \b
        RES:      Grid resolution in degrees.
        HEIGHT:   Number of vertical grid cells for given level
        HEIGHT_0: Number of vertical grid cells at lowest resolution level.

    Let WIDTH and HEIGHT be the number of horizontal and vertical grid cells
    of a global grid at a certain LEVEL with WIDTH * RES = 360 and HEIGHT * RES = 180, then
    we also force HEIGHT = TILE * 2 ^ LEVEL.
   """


cli.add_command(list_resolutions)
cli.add_command(list_levels)
cli.add_command(adjust_box)
