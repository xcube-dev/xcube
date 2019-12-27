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

from xcube.cli.common import parse_cli_sequence
from xcube.constants import FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4, FORMAT_NAME_MEM

OUTPUT_FORMAT_NAMES = [FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4, FORMAT_NAME_MEM]

DEFAULT_OUTPUT_PATH = 'out.zarr'
DEFAULT_XY_NAMES = 'lon,lat'
DEFAULT_DELTA = 1e-5
DEFAULT_DRY_RUN = False


# noinspection PyShadowingBuiltins
@click.command(name='reproject', hidden=True)
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
              help='Output size in pixels using format "WIDTH,HEIGHT", e.g. "512,512". '
                   'If omitted, a size will be computed so spatial resolution of INPUT is preserved.')
@click.option('--point', '-p', 'output_point', metavar='POINT',
              help='Output spatial coordinates of the point referring to pixel col=0.5,row=0.5 '
                   'using format "LON,LAT" or "X,Y", e.g. "1.2,53.5". '
                   'If omitted, the default reference point will the INPUT\'s minimum spatial coordinates.')
@click.option('--res', '-r', 'output_res', type=float,
              help='Output spatial resolution. '
                   'If omitted, the default resolution will be close to the spatial resolution of INPUT.')
@click.option('--delta', '-d', type=float, default=DEFAULT_DELTA,
              help='Relative maximum delta for detection whether a '
                   'target pixel center is within a source pixel\'s boundary.')
@click.option('--dry-run', default=DEFAULT_DRY_RUN, is_flag=True,
              help='Just read and process INPUT, but don\'t produce any outputs.')
def reproject(dataset: str,
              xy_var_names: str = None,
              var_names: str = None,
              output_path: str = None,
              output_format: str = None,
              output_size: str = None,
              output_point: str = None,
              output_res: float = None,
              delta: float = DEFAULT_DELTA,
              dry_run: bool = DEFAULT_DRY_RUN):
    """
    Reproject a dataset using its per-pixel geo-locations.
    """

    input_path = dataset

    def positive_int(v):
        if v <= 0:
            raise ValueError('must be positive')

    xy_var_names = parse_cli_sequence(xy_var_names, metavar='VARIABLES', num_items=2, item_plural_name='names',
                                      error_type=click.ClickException)
    var_names = parse_cli_sequence(var_names, metavar='VARIABLES', item_plural_name='names',
                                   error_type=click.ClickException)
    output_size = parse_cli_sequence(output_size, metavar='SIZE', num_items=2, item_plural_name='names',
                                     item_parser=int,
                                     item_validator=positive_int, error_type=click.ClickException)
    output_point = parse_cli_sequence(output_point, metavar='POINT', num_items=2, item_plural_name='coordinates',
                                      item_parser=float, error_type=click.ClickException)

    # noinspection PyBroadException
    _reproject(input_path,
               xy_var_names,
               var_names,
               output_path,
               output_format,
               output_size,
               output_point,
               output_res,
               delta,
               dry_run=dry_run,
               monitor=print)

    return 0


def _reproject(input_path: str,
               xy_names: Optional[Tuple[str, str]],
               var_names: Optional[Sequence[str]],
               output_path: str,
               output_format: Optional[str],
               output_size: Optional[Tuple[int, int]],
               output_point: Optional[Tuple[float, float]],
               output_res: Optional[float],
               delta: float,
               dry_run: bool,
               monitor):
    from xcube.core.dsio import guess_dataset_format
    from xcube.core.dsio import open_dataset
    from xcube.core.dsio import write_dataset
    from xcube.core.geocoded import reproject_dataset
    from xcube.core.geocoded import ImageGeom

    if not output_format:
        output_format = guess_dataset_format(output_path)

    output_geom = None
    if output_size is not None and output_point is not None and output_res is not None:
        output_geom = ImageGeom()
    elif output_size is not None or output_point is not None or output_res is not None:
        raise click.ClickException('SIZE, POINT, and RES must all be given or none of them.')

    monitor(f'Opening dataset from {input_path!r}...')

    if _is_s3_olci_path(input_path):
        src_ds = _open_s3_olci(input_path)
    else:
        src_ds = open_dataset(input_path)

    monitor('Reprojecting...')
    reproj_ds = reproject_dataset(src_ds,
                                  x_name=xy_names[0] if xy_names else None,
                                  y_name=xy_names[1] if xy_names else None,
                                  var_names=var_names,
                                  output_geom=output_geom,
                                  delta=delta)

    if reproj_ds is None:
        monitor(f'Dataset {input_path} does not seem to have an intersection with bounding box')
        return

    monitor(f'Writing reprojected dataset to {output_path!r}...')
    if not dry_run:
        write_dataset(reproj_ds, output_path, output_format)
    monitor(f'Done.')


def _open_s3_olci(input_path):
    import os
    import xarray as xr

    x_name = 'longitude'
    y_name = 'latitude'
    data_vars = {}
    geo_vars_file_name = 'geo_coordinates.nc'
    file_names = set(file_name for file_name in os.listdir(input_path) if file_name.endswith('.nc'))
    if geo_vars_file_name not in file_names:
        raise ValueError(f'missing file {geo_vars_file_name!r} in {input_path}')
    file_names.remove(geo_vars_file_name)
    geo_vars_path = os.path.join(input_path, geo_vars_file_name)
    with xr.open_dataset(geo_vars_path) as geo_ds:
        if x_name not in geo_ds:
            raise ValueError(f'variable {x_name!r} not found in {geo_vars_path}')
        if y_name not in geo_ds:
            raise ValueError(f'variable {y_name!r} not found in {geo_vars_path}')
        x_var = geo_ds[x_name]
        y_var = geo_ds[y_name]
        if x_var.ndim != 2:
            raise ValueError(f'variable {x_name!r} must have two dimensions')
        if y_var.ndim != x_var.ndim \
                or y_var.shape != x_var.shape \
                or y_var.dims != x_var.dims:
            raise ValueError(f'variable {y_name!r} must have same shape and dimensions as {x_name!r}')
        data_vars.update({x_name: x_var, y_name: y_var})
    for file_name in file_names:
        with xr.open_dataset(os.path.join(input_path, file_name)) as ds:
            for var_name, var in ds.data_vars.items():
                if var.ndim >= 2 \
                        and var.shape[-2:] == x_var.shape \
                        and var.dims[-2:] == x_var.dims:
                    data_vars.update({var_name: var})
    return xr.Dataset(data_vars)


def _is_s3_olci_path(path: str):
    import os
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, 'geo_coordinates.nc'))
