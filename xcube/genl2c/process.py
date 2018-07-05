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
import traceback
from typing import Sequence, Callable, Tuple, Set, Any, Dict, Optional

from .inputprocessor import InputProcessor
from ..io import rimraf, get_default_dataset_io_registry, DatasetIO
from ..reproject import reproject_to_wgs84
from ..timedim import add_time_coords
from ..utils import select_variables

__import__('xcube.plugin')


def generate_l2c_cube(src_files: Sequence[str],
                      src_type: str,
                      dst_size: Tuple[int, int],
                      dst_region: Tuple[float, float, float, float],
                      dst_var_names: Optional[Set[str]],
                      dst_metadata: Optional[Dict[str, Any]],
                      dst_resample_alg_name: str,
                      dst_dir: str,
                      dst_name: str,
                      dst_format: str = 'netcdf4',
                      dst_append: bool = False,
                      dry_run: bool = False,
                      monitor: Callable[..., None] = None) -> Tuple[Optional[str], bool]:
    dataset_io_registry = get_default_dataset_io_registry()

    input_processor = dataset_io_registry.find(src_type)
    if not input_processor:
        raise ValueError(f'unknown input type {src_type!r}')

    output_writer = dataset_io_registry.find(dst_format, modes={'w', 'a'} if dst_append else {'w'})
    if not output_writer:
        raise ValueError(f'unknown output format {dst_format!r}')

    if monitor is None:
        # noinspection PyUnusedLocal
        def monitor(*args):
            pass

    src_files = sorted([input_file for f in src_files for input_file in glob.glob(f, recursive=True)])

    if not dry_run:
        os.makedirs(dst_dir, exist_ok=True)

    dst_path = None
    status = False

    ds_count = len(src_files)
    ds_index = 0
    for src_file in src_files:
        monitor(f'processing dataset {ds_index + 1} of {ds_count}: {src_file!r}...')
        # noinspection PyTypeChecker
        dst_path, status = _process_l2_input(input_processor,
                                             output_writer,
                                             src_file,
                                             dst_size,
                                             dst_region,
                                             dst_var_names,
                                             dst_metadata,
                                             dst_resample_alg_name,
                                             dst_dir,
                                             dst_name,
                                             dst_append,
                                             dry_run,
                                             monitor)
        if status:
            ds_index += 1

    monitor(f'{ds_index} of {ds_count} datasets processed successfully, '
            f'{ds_count - ds_index} were dropped due to errors')

    return dst_path, status


def _process_l2_input(input_processor: InputProcessor,
                      output_writer: DatasetIO,
                      src_file: str,
                      dst_size: Tuple[int, int],
                      dst_region: Tuple[float, float, float, float],
                      dst_var_names: Set[str],
                      dst_metadata: Optional[Dict[str, Any]],
                      dst_resample_alg_name: str,
                      dst_dir: str,
                      dst_name: str,
                      dst_append: bool,
                      dry_run: bool = False,
                      monitor: Callable[..., None] = None) -> Tuple[Optional[str], bool]:
    basename = os.path.basename(src_file)
    basename, ext = basename.rsplit('.', 1) if '.' in basename else (basename, None)

    dst_name = dst_name.format(INPUT_FILE=basename)
    output_basename = dst_name + '.' + output_writer.ext
    output_path = os.path.join(dst_dir, output_basename)

    monitor('reading...')
    # noinspection PyBroadException
    try:
        dataset = input_processor.read(src_file)
    except Exception as e:
        monitor(f'ERROR: cannot read input: {e}: skipping...')
        traceback.print_exc()
        return None, False

    src_dataset = dataset
    dataset = select_variables(dataset, dst_var_names)

    if dst_metadata and 'data_variables' in dst_metadata:
        dst_vars = dst_metadata['data_variables']
    else:
        dst_vars = {}

    try:
        monitor('pre-processing...')
        dataset = input_processor.pre_process(dataset)
        reprojection_info = input_processor.get_reprojection_info(src_dataset)
        if reprojection_info is not None:
            monitor('reprojecting...')
            dataset = reproject_to_wgs84(dataset,
                                         src_xy_var_names=reprojection_info.xy_var_names,
                                         src_xy_tp_var_names=reprojection_info.xy_tp_var_names,
                                         src_xy_crs=reprojection_info.xy_crs,
                                         src_xy_gcp_step=reprojection_info.xy_gcp_step or 1,
                                         src_xy_tp_gcp_step=reprojection_info.xy_tp_gcp_step or 1,
                                         dst_size=dst_size,
                                         dst_region=dst_region,
                                         dst_vars=dst_vars,
                                         dst_resample_alg_name=dst_resample_alg_name,
                                         include_non_spatial_vars=False)
        time_range = input_processor.get_time_range(src_dataset)
        if time_range is not None:
            monitor('adding-time coordinates...')
            dataset = add_time_coords(dataset, time_range)
        monitor('post-processing...')
        dataset = input_processor.post_process(dataset)
    except RuntimeError as e:
        monitor(f'ERROR: during reprojection to WGS84: {e}: skipping it...')
        traceback.print_exc()
        return output_path, False

    if dst_metadata and 'global_attributes' in dst_metadata:
        global_attributes = dst_metadata['global_attributes']
        dataset.attrs.clear()
        dataset.attrs.update(global_attributes)

    if not dry_run:
        if dst_append and os.path.exists(output_path):
            monitor(f'appending to {output_path}...')
            output_writer.append(dataset, output_path)
        else:
            rimraf(output_path)
            monitor(f'writing to {output_path}...')
            output_writer.write(dataset, output_path)

    dataset.close()

    return output_path, True
