# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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
from typing import Sequence, Optional, Tuple

import click

from xcube.cli.common import parse_cli_sequence, assert_positive_int_item
from xcube.constants import FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4, FORMAT_NAME_MEM

OUTPUT_FORMAT_NAMES = [FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4, FORMAT_NAME_MEM]

DEFAULT_OUTPUT_PATH = 'out.zarr'
DEFAULT_XY_NAMES = 'lon,lat'
DEFAULT_DELTA = 1e-5
DEFAULT_DRY_RUN = False


# noinspection PyShadowingBuiltins
@click.command(name='rectify', hidden=True)
@click.argument('dataset', metavar='INPUT')
@click.option('--xy-vars', 'xy_var_names',
              help=f'Comma-separated names of variables providing x any y coordinates. '
                   f'If omitted, names will be guessed from available variables in INPUT, e.g. "lon,lat".')
@click.option('--var', '-v', 'var_names', multiple=True, metavar='VARIABLES',
              help="Comma-separated list of names of variables to be included or multiple options may be given. "
                   "If omitted, all variables in INPUT will be reprojected.")
@click.option('--output', '-o', 'output_path', metavar='OUTPUT',
              default=DEFAULT_OUTPUT_PATH,
              help=f"Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}.")
@click.option('--format', '-f', 'output_format', metavar='FORMAT',
              type=click.Choice(OUTPUT_FORMAT_NAMES),
              help="Output format. "
                   "If omitted, format will be guessed from OUTPUT.")
@click.option('--size', '-s', 'output_size', metavar='SIZE',
              help='Output size in pixels using format "WIDTH,HEIGHT", e.g. "2048,1024". '
                   'If omitted, a size will be computed so spatial resolution of INPUT is preserved.')
@click.option('--tile_size', '-t', 'output_tile_size', metavar='TILE_SIZE',
              help='Output tile size in pixels using format "WIDTH,HEIGHT", e.g. "512,512". '
                   'If given, the output will be chunked w.r.t. TILE_SIZE. Otherwise the output will NOT be chunked.')
@click.option('--point', '-p', 'output_point', metavar='POINT',
              help='Output spatial coordinates of the point referring to pixel col=0.5,row=0.5 '
                   'using format "LON,LAT" or "X,Y", e.g. "1.2,53.5". '
                   'If omitted, the default reference point will the INPUT\'s minimum spatial coordinates.')
@click.option('--res', '-r', 'output_res', type=float,
              help='Output spatial resolution in decimal degrees. '
                   'If omitted, the default resolution will be close to the spatial resolution of INPUT.')
@click.option('--delta', '-d', type=float, default=DEFAULT_DELTA,
              help='Relative maximum delta for detection whether a '
                   'target pixel center is within a source pixel\'s boundary.')
@click.option('--dry-run', default=DEFAULT_DRY_RUN, is_flag=True,
              help='Just read and process INPUT, but don\'t produce any outputs.')
def rectify(dataset: str,
            xy_var_names: str = None,
            var_names: str = None,
            output_path: str = None,
            output_format: str = None,
            output_size: str = None,
            output_tile_size: str = None,
            output_point: str = None,
            output_res: float = None,
            delta: float = DEFAULT_DELTA,
            dry_run: bool = DEFAULT_DRY_RUN):
    """
    Rectify a dataset to WGS-84 using its per-pixel geo-locations.
    """

    input_path = dataset

    xy_var_names = parse_cli_sequence(xy_var_names,
                                      metavar='VARIABLES', num_items=2, item_plural_name='names')
    var_names = parse_cli_sequence(var_names,
                                   metavar='VARIABLES', item_plural_name='names')
    output_size = parse_cli_sequence(output_size,
                                     metavar='SIZE', num_items=2, item_plural_name='sizes',
                                     item_parser=int, item_validator=assert_positive_int_item)
    output_tile_size = parse_cli_sequence(output_tile_size,
                                          metavar='TILE_SIZE', num_items=2, item_plural_name='tile sizes',
                                          item_parser=int, item_validator=assert_positive_int_item)
    output_point = parse_cli_sequence(output_point,
                                      metavar='POINT', num_items=2, item_plural_name='coordinates',
                                      item_parser=float)

    # noinspection PyBroadException
    _rectify(input_path,
             xy_var_names,
             None if len(var_names) == 0 else var_names,
             output_path,
             output_format,
             output_size,
             output_tile_size,
             output_point,
             output_res,
             delta,
             dry_run=dry_run,
             monitor=print)

    return 0


def _rectify(input_path: str,
             xy_names: Optional[Tuple[str, str]],
             var_names: Optional[Sequence[str]],
             output_path: str,
             output_format: Optional[str],
             output_size: Optional[Tuple[int, int]],
             output_tile_size: Optional[Tuple[int, int]],
             output_point: Optional[Tuple[float, float]],
             output_res: Optional[float],
             delta: float,
             dry_run: bool,
             monitor):
    from xcube.core.dsio import guess_dataset_format
    from xcube.core.dsio import open_dataset
    from xcube.core.dsio import write_dataset
    from xcube.core.rectify import rectify_dataset
    from xcube.core.rectify import ImageGeom
    from xcube.core.sentinel3 import is_sentinel3_product
    from xcube.core.sentinel3 import open_sentinel3_product

    if not output_format:
        output_format = guess_dataset_format(output_path)

    output_geom = None
    if output_size is not None and output_point is not None and output_res is not None:
        output_geom = ImageGeom(size=output_size, x_min=output_point[0], y_min=output_point[1], xy_res=output_res)
    elif output_size is not None or output_point is not None or output_res is not None:
        raise click.ClickException('SIZE, POINT, and RES must all be given or none of them.')

    monitor(f'Opening dataset from {input_path!r}...')

    if is_sentinel3_product(input_path):
        src_ds = open_sentinel3_product(input_path)
    else:
        src_ds = open_dataset(input_path)

    monitor('Rectifying...')
    reproj_ds = rectify_dataset(src_ds,
                                xy_names=xy_names,
                                var_names=var_names,
                                output_geom=output_geom,
                                tile_size=output_tile_size,
                                uv_delta=delta)

    if reproj_ds is None:
        monitor(f'Dataset {input_path} does not seem to have an intersection with bounding box')
        return

    monitor(f'Writing rectified dataset to {output_path!r}...')
    if not dry_run:
        write_dataset(reproj_ds, output_path, output_format)

    monitor(f'Done.')
