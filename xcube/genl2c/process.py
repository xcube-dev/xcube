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
from typing import Sequence, Callable, Tuple, Any, Dict, Optional, List, Union

from .defaults import DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_NAME, DEFAULT_OUTPUT_SIZE, \
    DEFAULT_OUTPUT_RESAMPLING, DEFAULT_OUTPUT_FORMAT
from .inputprocessor import InputProcessor
from ..expression import compute_dataset
from ..config import flatten_dict
from ..io import rimraf, get_default_dataset_io_registry, DatasetIO
from ..reproject import reproject_to_wgs84
from ..timedim import add_time_coords
from ..utils import select_variables, update_variable_props

__import__('xcube.plugin')


def generate_l2c_cube(input_files: Sequence[str] = None,
                      input_type: str = None,
                      output_region: Tuple[float, float, float, float] = None,
                      output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
                      output_resampling: str = DEFAULT_OUTPUT_RESAMPLING,
                      output_dir: str = DEFAULT_OUTPUT_DIR,
                      output_name: str = DEFAULT_OUTPUT_NAME,
                      output_format: str = DEFAULT_OUTPUT_FORMAT,
                      output_metadata: Dict[str, Any] = None,
                      output_variables: List[Union[str, Dict[str, str], Dict[str, Dict[str, Any]]]] = None,
                      processed_variables: List[Union[str, Dict[str, Dict[str, Any]]]] = None,
                      append_mode: bool = False,
                      dry_run: bool = False,
                      monitor: Callable[..., None] = None) -> Tuple[Optional[str], bool]:
    dataset_io_registry = get_default_dataset_io_registry()

    input_processor = dataset_io_registry.find(input_type)
    if not input_processor:
        raise ValueError(f'unknown input type {input_type!r}')

    output_writer = dataset_io_registry.find(output_format, modes={'w', 'a'} if append_mode else {'w'})
    if not output_writer:
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
        output_path, status = _process_l2_input(input_processor,
                                                output_writer,
                                                input_file,
                                                output_size,
                                                output_region,
                                                output_resampling,
                                                output_dir,
                                                output_name,
                                                output_metadata,
                                                output_variables,
                                                processed_variables,
                                                append_mode,
                                                dry_run,
                                                monitor)
        if status:
            ds_index += 1

    monitor(f'{ds_index} of {ds_count} datasets processed successfully, '
            f'{ds_count - ds_index} were dropped due to errors')

    return output_path, status


def _process_l2_input(input_processor: InputProcessor,
                      output_writer: DatasetIO,
                      input_file: str,
                      output_size: Tuple[int, int],
                      output_region: Tuple[float, float, float, float],
                      output_resampling: str,
                      output_dir: str,
                      output_name: str,
                      output_metadata: Dict[str, Any] = None,
                      output_variables: List[Union[str, Dict[str, str], Dict[str, Dict[str, Any]]]] = None,
                      processed_variables: List[Union[str, Dict[str, Dict[str, Any]]]] = None,
                      append_mode: bool = False,
                      dry_run: bool = False,
                      monitor: Callable[..., None] = None) -> Tuple[Optional[str], bool]:
    basename = os.path.basename(input_file)
    basename, ext = basename.rsplit('.', 1) if '.' in basename else (basename, None)

    output_name = output_name.format(INPUT_FILE=basename)
    output_basename = output_name + '.' + output_writer.ext
    output_path = os.path.join(output_dir, output_basename)

    monitor('reading dataset...')
    # noinspection PyBroadException
    try:
        input_dataset = input_processor.read(input_file)
    except Exception as e:
        monitor(f'ERROR: cannot read input: {e}: skipping...')
        traceback.print_exc()
        return None, False

    reprojection_info = input_processor.get_reprojection_info(input_dataset)
    time_range = input_processor.get_time_range(input_dataset)

    selected_variables = list(output_variables)
    if reprojection_info.xy_var_names:
        selected_variables.extend(reprojection_info.xy_var_names)
    if reprojection_info.xy_tp_var_names:
        selected_variables.extend(reprojection_info.xy_tp_var_names)

    try:
        monitor('pre-processing dataset...')
        preprocessed_dataset = input_processor.pre_process(input_dataset)
        monitor('computing variables...')
        computed_dataset = compute_dataset(preprocessed_dataset, processed_variables=processed_variables)
        monitor('selecting variables...')
        subset_dataset = select_variables(computed_dataset, selected_variables)
        if reprojection_info is not None:
            monitor('reprojecting dataset...')
            reprojected_dataset = reproject_to_wgs84(subset_dataset,
                                                     src_xy_var_names=reprojection_info.xy_var_names,
                                                     src_xy_tp_var_names=reprojection_info.xy_tp_var_names,
                                                     src_xy_crs=reprojection_info.xy_crs,
                                                     src_xy_gcp_step=reprojection_info.xy_gcp_step or 1,
                                                     src_xy_tp_gcp_step=reprojection_info.xy_tp_gcp_step or 1,
                                                     dst_size=output_size,
                                                     dst_region=output_region,
                                                     dst_resampling=output_resampling,
                                                     include_non_spatial_vars=False)
        else:
            reprojected_dataset = subset_dataset
        if time_range is not None:
            monitor('adding time coordinates...')
            reprojected_dataset = add_time_coords(reprojected_dataset, time_range)
        monitor('updating variable properties...')
        reprojected_dataset = update_variable_props(reprojected_dataset, output_variables)
        monitor('post-processing dataset...')
        post_processed_dataset = input_processor.post_process(reprojected_dataset)
    except RuntimeError as e:
        monitor(f'ERROR: during reprojection to WGS84: {e}: skipping it...')
        traceback.print_exc()
        return output_path, False

    if output_metadata:
        post_processed_dataset.attrs.update(flatten_dict(output_metadata))

    if not dry_run:
        if append_mode and os.path.exists(output_path):
            monitor(f'appending to {output_path}...')
            output_writer.append(post_processed_dataset, output_path)
        else:
            rimraf(output_path)
            monitor(f'writing to {output_path}...')
            output_writer.write(post_processed_dataset, output_path)

    input_dataset.close()

    return output_path, True
