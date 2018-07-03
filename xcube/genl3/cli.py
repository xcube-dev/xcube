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

import argparse
import glob
import os
import sys
from typing import List, Optional

import xarray as xr

from xcube.metadata import load_metadata_yaml
from xcube.version import version

__import__('xcube.plugin')

DEFAULT_OUTPUT_DIR = '.'
DEFAULT_OUTPUT_PATTERN = 'L3_{INPUT_FILE}'
DEFAULT_OUTPUT_FORMAT = 'zarr'
DEFAULT_OUTPUT_RESAMPLING = 'nearest'
DEFAULT_OUTPUT_FREQUENCY = '1D'


def main(args: Optional[List[str]] = None):
    """
    Generate L2C data cubes from L2 data products.
    """
    output_format_names = ['zarr', 'nc']
    resampling_algs = ['all', 'any', 'argmin', 'argmax', 'count', 'first', 'last', 'max', 'mean', 'median', 'min',
                       'backfill', 'bfill', 'ffill', 'interpolate', 'nearest', 'pad']

    parser = argparse.ArgumentParser(description='Generate Level-3 data cube from Level-2C data cube.')
    parser.add_argument('--version', '-V', action='version', version=version)
    parser.add_argument('--dir', '-d', dest='output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory. Defaults to {DEFAULT_OUTPUT_DIR!r}')
    parser.add_argument('--name', '-n', dest='output_name', default=DEFAULT_OUTPUT_PATTERN,
                        help=f'Output filename pattern. Defaults to {DEFAULT_OUTPUT_PATTERN!r}.')
    parser.add_argument('--format', '-f', dest='output_format',
                        default=output_format_names[0], choices=output_format_names,
                        help=f'Output format. Defaults to {output_format_names[0]!r}.')
    parser.add_argument('--variables', '-v', dest='output_variables',
                        help='Variables to be included in output. '
                             'Comma-separated list of names which may contain wildcard characters "*" and "?".')
    parser.add_argument('--resampling', dest='output_resampling', choices=resampling_algs,
                        help='Fallback resampling algorithm to be used for all variables.'
                             f'Defaults to {DEFAULT_OUTPUT_RESAMPLING!r}.')
    parser.add_argument('--frequency', dest='output_frequency',
                        help='Aggregation frequency.'
                             f'Defaults to {DEFAULT_OUTPUT_FREQUENCY!r}.')
    parser.add_argument('--meta-file', '-m', dest='output_meta_file',
                        help='File containing cube-level, CF-compliant metadata in YAML format.')
    parser.add_argument('--dry-run', default=False, action='store_true',
                        help='Just read and process inputs, but don\'t produce any outputs.')
    parser.add_argument('input_file', metavar='INPUT_FILE',
                        help="The input file or directory which must be a Level-2C cube.")

    try:
        arg_obj = parser.parse_args(args or sys.argv[1:])
    except SystemExit as e:
        return int(str(e))

    input_file = arg_obj.input_file
    output_dir = arg_obj.output_dir or DEFAULT_OUTPUT_DIR
    output_name = arg_obj.output_name or DEFAULT_OUTPUT_PATTERN
    output_format = arg_obj.output_format or DEFAULT_OUTPUT_FORMAT
    output_variables = arg_obj.output_variables
    output_resampling = arg_obj.output_resampling or DEFAULT_OUTPUT_RESAMPLING
    output_frequency = arg_obj.output_frequency or DEFAULT_OUTPUT_FREQUENCY
    output_meta_file = arg_obj.output_meta_file
    dry_run = arg_obj.dry_run

    if output_variables:
        try:
            output_variables = set(map(lambda c: str(c).strip(), output_variables.split(',')))
        except ValueError:
            output_variables = None
        if output_variables is not None \
                and next(iter(True for var_name in output_variables if var_name == ''), False):
            output_variables = None
        if output_variables is None or len(output_variables) == 0:
            print(f'error: invalid variables {arg_obj.output_variables!r}')
            return 2

    if output_meta_file:
        try:
            with open(output_meta_file) as stream:
                output_metadata = load_metadata_yaml(stream)
            print(f'loaded metadata from file {arg_obj.output_meta_file!r}')
        except OSError as e:
            print(f'error: failed loading metadata file {arg_obj.output_meta_file!r}: {e}')
            return 2
    else:
        output_metadata = None

    process_l2c_cube(input_file,
                     output_variables,
                     output_metadata,
                     output_resampling,
                     output_dir,
                     output_name,
                     output_format,
                     output_frequency,
                     dry_run=dry_run,
                     monitor=print)
    return 0


def process_l2c_cube(input_file: str,
                     output_variables=None,
                     output_metadata=None,
                     output_resampling=DEFAULT_OUTPUT_RESAMPLING,
                     output_dir=DEFAULT_OUTPUT_DIR,
                     output_name=DEFAULT_OUTPUT_PATTERN,
                     output_format=DEFAULT_OUTPUT_FORMAT,
                     output_frequency=DEFAULT_OUTPUT_FREQUENCY,
                     dry_run=False,
                     monitor=None):
    input_file_name = os.path.basename(input_file)
    output_path = os.path.join(output_dir, output_name.format(input_file_name) + '.' + output_format)

    monitor(f'Reading L2C cube from {input_file!r}...')
    if os.path.isdir(input_file):
        if input_file_name.endswith('.zarr'):
            ds = xr.open_zarr(input_file)
        else:
            ds = xr.open_mfdataset(glob.glob(os.path.join(input_file, '**', '*.nc'), recursive=True))
    else:
        if input_file_name.endswith('.zarr.zip'):
            ds = xr.open_zarr(input_file)
        else:
            ds = xr.open_dataset(input_file)

    if output_variables:
        ds = ds.drop(*set(ds.data_vars).difference(output_variables))

    monitor('Resampling...')
    resampled_ds = ds.resample(skipna=True, closed='left', label='left', keep_attrs=True,
                               how=output_resampling, time=output_frequency)

    # TODO: add time_bnds to resampled_ds

    time_coverage_start = '%s' % ds.time[0]
    time_coverage_end = '%s' % ds.time[-1]

    resampled_ds.attrs.update(output_metadata)
    resampled_ds.attrs.update(time_coverage_start=time_coverage_start,
                              time_coverage_end=time_coverage_end)

    monitor(f'Writing L3 cube to {output_path!r}...')
    if not dry_run:
        if output_format == 'zarr':
            resampled_ds.to_zarr(output_path)
        elif output_format == 'nc':
            resampled_ds.to_netcdf(output_path)

    monitor(f'Done.')


if __name__ == '__main__':
    sys.exit(status=main())
