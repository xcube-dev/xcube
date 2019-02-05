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

import argparse
import glob
import os
import sys
import traceback
from typing import List, Optional

import xarray as xr
import yaml

from xcube.api.restime import resample_in_time, RESAMPLING_METHODS
from xcube.version import version

# TODO (forman): use DatasetIO registry!
OUTPUT_FORMAT_NAMES = ['zarr', 'nc']

DEFAULT_OUTPUT_DIR = '.'
DEFAULT_OUTPUT_PATTERN = 'L3_{INPUT_FILE}'
DEFAULT_OUTPUT_FORMAT = 'zarr'
DEFAULT_OUTPUT_RESAMPLING_METHOD = 'nearest'
DEFAULT_OUTPUT_FREQUENCY = '1D'


def main(args: Optional[List[str]] = None):
    """
    Data cube resampling in time.
    """

    parser = argparse.ArgumentParser(description='Data cube resampling in time.')
    parser.add_argument('--version', '-V', action='version', version=version)
    parser.add_argument('--dir', '-d', dest='output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory. Defaults to {DEFAULT_OUTPUT_DIR!r}')
    parser.add_argument('--name', '-n', dest='output_name', default=DEFAULT_OUTPUT_PATTERN,
                        help=f'Output filename pattern. Defaults to {DEFAULT_OUTPUT_PATTERN!r}.')
    parser.add_argument('--format', '-f', dest='output_format',
                        default=DEFAULT_OUTPUT_FORMAT, choices=OUTPUT_FORMAT_NAMES,
                        help=f'Output format. Defaults to {DEFAULT_OUTPUT_FORMAT!r}.')
    parser.add_argument('--variables', '-v', dest='output_variables',
                        help='Variables to be included in output. '
                             'Comma-separated list of names which may contain wildcard characters "*" and "?".')
    parser.add_argument('--resampling', dest='output_resampling', choices=RESAMPLING_METHODS,
                        help='Temporal resampling method. Use format "<count><offset>"'
                             'where <offset> is one of {H, D, W, M, Q, Y}'
                        f'Defaults to {DEFAULT_OUTPUT_RESAMPLING_METHOD!r}.')
    parser.add_argument('--frequency', dest='output_frequency',
                        help='Temporal aggregation frequency.'
                        f'Defaults to {DEFAULT_OUTPUT_FREQUENCY!r}.')
    parser.add_argument('--meta-file', '-m', dest='output_meta_file',
                        help='File containing cube-level, CF-compliant metadata in YAML format.')
    parser.add_argument('--dry-run', default=False, action='store_true',
                        help='Just read and process inputs, but don\'t produce any outputs.')
    parser.add_argument('--traceback', dest='traceback_mode', default=False, action='store_true',
                        help='On error, print Python traceback.')
    parser.add_argument('input_file', metavar='INPUT_FILE',
                        help="The input file or directory which must be a Level-2C cube.")

    try:
        arg_obj = parser.parse_args(args or sys.argv[1:])
    except SystemExit as e:
        return int(str(e))

    config_file = arg_obj.output_meta_file
    input_file = arg_obj.input_file
    output_dir = arg_obj.output_dir or DEFAULT_OUTPUT_DIR
    output_name = arg_obj.output_name or DEFAULT_OUTPUT_PATTERN
    output_format = arg_obj.output_format or DEFAULT_OUTPUT_FORMAT
    output_variables = arg_obj.output_variables
    output_resampling = arg_obj.output_resampling or DEFAULT_OUTPUT_RESAMPLING_METHOD
    output_frequency = arg_obj.output_frequency or DEFAULT_OUTPUT_FREQUENCY
    traceback_mode = arg_obj.traceback_mode
    dry_run = arg_obj.dry_run

    if config_file:
        try:
            with open(config_file) as stream:
                config = yaml.load(stream)
            print(f'loaded configuration from {config_file!r}')
        except OSError as e:
            print(f'error: failed loading configuration from {config_file!r}: {e}')
            if traceback_mode:
                traceback.print_exc()
            return 2
    else:
        config = {}

    if input_file:
        config['input_file'] = input_file
    if output_dir:
        config['output_dir'] = output_dir
    if output_name:
        config['output_name'] = output_name
    if output_format:
        config['output_format'] = output_format
    if output_resampling:
        config['output_resampling'] = output_resampling
    if output_frequency:
        config['output_frequency'] = output_frequency
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
        # TODO: instead filter list from config
        config['output_variables'] = output_variables

    # noinspection PyBroadException
    try:
        _resample_in_time(*config,
                          dry_run=dry_run,
                          monitor=print)
    except BaseException as e:
        print(f'error: {e}')
        if traceback_mode:
            traceback.print_exc()
        return 2

    return 0


if __name__ == '__main__':
    sys.exit(status=main())


def _resample_in_time(input_file: str,
                      output_variables=None,
                      output_metadata=None,
                      output_dir=DEFAULT_OUTPUT_DIR,
                      output_name=DEFAULT_OUTPUT_PATTERN,
                      output_format=DEFAULT_OUTPUT_FORMAT,
                      output_resampling=DEFAULT_OUTPUT_RESAMPLING_METHOD,
                      output_frequency=DEFAULT_OUTPUT_FREQUENCY,
                      dry_run=False,
                      monitor=None):
    input_file_name = os.path.basename(input_file)
    output_path = os.path.join(output_dir, output_name.format(INPUT_FILE=input_file_name) + '.' + output_format)

    monitor(f'Reading L2C cube from {input_file!r}...')
    ds = _read_dataset(input_file)

    monitor('Resampling...')
    resampled_ds = resample_in_time(ds, output_frequency, output_resampling, output_variables, output_metadata)

    monitor(f'Writing L3 cube to {output_path!r}...')
    if not dry_run:
        _write_dataset(resampled_ds, output_path, output_format)

    monitor(f'Done.')


def _read_dataset(input_file):
    input_file_name = os.path.basename(input_file)
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
    return ds


def _write_dataset(ds, output_path, output_format):
    if output_format == 'zarr':
        ds.to_zarr(output_path)
    elif output_format == 'nc':
        ds.to_netcdf(output_path)
