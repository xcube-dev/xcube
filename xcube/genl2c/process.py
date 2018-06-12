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
from ..utils import select_variables

__import__('xcube.plugin')


def process_inputs(input_files: Sequence[str],
                   input_type: str,
                   dst_size: Tuple[int, int],
                   dst_region: Tuple[float, float, float, float],
                   dst_variables: Set[str],
                   dst_metadata: Optional[Dict[str, Any]],
                   output_dir: str,
                   output_name: str,
                   output_format: str = 'netcdf4',
                   append: bool = False,
                   dry_run: bool = False,
                   monitor: Callable[..., None] = None) -> Tuple[Optional[str], bool]:
    dataset_io_registry = get_default_dataset_io_registry()

    input_processor = dataset_io_registry.find(input_type)
    if not input_processor:
        raise ValueError(f'unknown input type {input_type!r}')

    dataset_writer = dataset_io_registry.find(output_format, modes={'w', 'a'} if append else {'w'})
    if not dataset_writer:
        raise ValueError(f'unknown output format {output_format!r}')

    if monitor is None:
        # noinspection PyUnusedLocal
        def monitor(*args):
            pass

    input_files = sorted([input_file for f in input_files for input_file in glob.glob(f, recursive=True)])

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    output_path = None
    status = False

    ds_count = len(input_files)
    ds_index = 0
    for input_file in input_files:
        monitor(f'processing dataset {ds_index + 1} of {ds_count}: {input_file!r}...')
        # noinspection PyTypeChecker
        output_path, status = process_input(input_file,
                                            input_processor,
                                            dst_size,
                                            dst_region,
                                            dst_variables,
                                            dst_metadata,
                                            output_dir,
                                            output_name,
                                            dataset_writer,
                                            append,
                                            dry_run,
                                            monitor)
        if status:
            ds_index += 1

    monitor(f'{ds_index} of {ds_count} datasets processed successfully, '
            f'{ds_count - ds_index} were dropped due to errors')

    return output_path, status


def process_input(input_file: str,
                  input_processor: InputProcessor,
                  dst_size: Tuple[int, int],
                  dst_region: Tuple[float, float, float, float],
                  dst_variables: Set[str],
                  dst_metadata: Dict[str, Any],
                  output_dir: str,
                  output_name: str,
                  dataset_writer: DatasetIO,
                  append: bool,
                  dry_run: bool = False,
                  monitor: Callable[..., None] = None) -> Tuple[Optional[str], bool]:
    basename = os.path.basename(input_file)
    basename, ext = basename.rsplit('.', 1) if '.' in basename else (basename, None)

    output_name = output_name.format(INPUT_FILE=basename)
    output_basename = output_name + '.' + dataset_writer.ext
    output_path = os.path.join(output_dir, output_basename)

    monitor('reading...')
    # noinspection PyBroadException
    try:
        dataset = input_processor.read(input_file)
    except Exception as e:
        monitor(f'ERROR: cannot read input: {e}: skipping...')
        traceback.print_exc()
        return None, False

    dataset = select_variables(dataset, dst_variables)

    try:
        monitor('pre-processing...')
        dataset = input_processor.pre_reproject(dataset)
        monitor('reprojecting...')
        input_info = input_processor.input_info
        dataset = reproject_to_wgs84(dataset,
                                     src_xy_var_names=input_info.xy_var_names,
                                     src_xy_tp_var_names=input_info.xy_tp_var_names,
                                     src_xy_crs=input_info.xy_crs,
                                     src_time_var_name=input_info.time_var_name,
                                     src_time_range_attr_names=input_info.time_range_attr_names,
                                     src_time_format=input_info.time_format,
                                     dst_size=dst_size,
                                     dst_region=dst_region,
                                     gcp_step=5,
                                     include_non_spatial_vars=False)
        monitor('post-processing...')
        dataset = input_processor.post_reproject(dataset)
    except RuntimeError as e:
        monitor(f'ERROR: during reprojection to WGS84: {e}: skipping it...')
        traceback.print_exc()
        return output_path, False

    if dst_metadata:
        dataset.attrs.clear()
        dataset.attrs.update(dst_metadata)

    if not dry_run:
        if append and os.path.exists(output_path):
            monitor(f'appending to {output_path}...')
            dataset_writer.append(dataset, output_path)
        else:
            rimraf(output_path)
            monitor(f'writing to {output_path}...')
            dataset_writer.write(dataset, output_path)

    dataset.close()

    return output_path, True
