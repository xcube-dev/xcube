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

import glob
import os
from typing import Dict, Any, Set

import xarray as xr

from xcube.genl3.defaults import DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_PATTERN, DEFAULT_OUTPUT_FORMAT, \
    DEFAULT_OUTPUT_RESAMPLING_METHOD, DEFAULT_OUTPUT_FREQUENCY
from xcube.utils import select_variables

OUTPUT_FORMAT_NAMES = ['zarr', 'nc']
RESAMPLING_METHODS = ['all', 'any', 'argmin', 'argmax', 'count', 'first', 'last', 'max', 'mean', 'median', 'min',
                      'backfill', 'bfill', 'ffill', 'interpolate', 'nearest', 'pad']


def generate_l3_cube(input_file: str,
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
    resampled_ds = resample(ds, output_frequency, output_resampling, output_variables, output_metadata)

    monitor(f'Writing L3 cube to {output_path!r}...')
    if not dry_run:
        _write_dataset(resampled_ds, output_path, output_format)

    monitor(f'Done.')


def resample(ds: xr.Dataset,
             frequency: str,
             resampling: str,
             var_name_patterns: Set[str] = None,
             metadata: Dict[str, Any] = None):
    if var_name_patterns:
        ds = select_variables(ds, var_name_patterns)

    resampler = ds.resample(skipna=True,
                            closed='left',
                            label='left',
                            keep_attrs=True,
                            time=frequency)

    resampling_method = getattr(resampler, resampling)
    resampled_ds = resampling_method('time')

    # TODO: add time_bnds to resampled_ds
    time_coverage_start = '%s' % ds.time[0]
    time_coverage_end = '%s' % ds.time[-1]

    resampled_ds.attrs.update(metadata or {})
    # TODO: add other time_coverage_ attributes
    resampled_ds.attrs.update(time_coverage_start=time_coverage_start,
                              time_coverage_end=time_coverage_end)
    return resampled_ds


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
