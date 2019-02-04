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

import cProfile
import glob
import io
import os
import pstats
import time
import traceback
from typing import Sequence, Callable, Tuple, Optional, Dict, Any

from .defaults import DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_NAME, DEFAULT_OUTPUT_SIZE, \
    DEFAULT_OUTPUT_RESAMPLING, DEFAULT_OUTPUT_WRITER
from .inputprocessor import InputProcessor, get_input_processor
from ..config import NameAnyDict, NameDictPairList, to_resolved_name_dict_pairs
from ..dsio import rimraf, DatasetIO, find_dataset_io
from ..dsutil import compute_dataset, select_variables, update_variable_props, add_time_coords, update_global_attributes
from ..reproject import reproject_to_wgs84

__import__('xcube.plugin')

_PROFILING_ON = False


def generate_l2c_cube(input_files: Sequence[str] = None,
                      input_processor: str = None,
                      input_processor_params: Dict = None,
                      input_reader: str = None,
                      input_reader_params: Dict[str, Any] = None,
                      output_region: Tuple[float, float, float, float] = None,
                      output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
                      output_resampling: str = DEFAULT_OUTPUT_RESAMPLING,
                      output_dir: str = DEFAULT_OUTPUT_DIR,
                      output_name: str = DEFAULT_OUTPUT_NAME,
                      output_writer: str = DEFAULT_OUTPUT_WRITER,
                      output_writer_params: Dict[str, Any] = None,
                      output_metadata: NameAnyDict = None,
                      output_variables: NameDictPairList = None,
                      processed_variables: NameDictPairList = None,
                      append_mode: bool = False,
                      dry_run: bool = False,
                      monitor: Callable[..., None] = None) -> Tuple[Optional[str], bool]:
    input_processor = get_input_processor(input_processor)
    if not input_processor:
        raise ValueError(f'unknown input_processor {input_processor!r}')

    if input_processor_params:
        try:
            input_processor.configure(**input_processor_params)
        except TypeError as e:
            raise ValueError(f'invalid input_processor_params {input_processor_params!r}') from e

    input_reader = find_dataset_io(input_reader or input_processor.input_reader)
    if not input_reader:
        raise ValueError(f'unknown input_reader {input_reader!r}')

    output_writer = find_dataset_io(output_writer or 'netcdf4', modes={'w', 'a'} if append_mode else {'w'})
    if not output_writer:
        raise ValueError(f'unknown output_writer {output_writer!r}')

    if monitor is None:
        # noinspection PyUnusedLocal
        def monitor(*args):
            pass

    input_files = sorted([input_file for f in input_files for input_file in glob.glob(f, recursive=True)])

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    effective_input_reader_params = dict(input_processor.input_reader_params or {})
    effective_input_reader_params.update(input_reader_params or {})

    effective_output_writer_params = output_writer_params or {}

    output_path = None
    status = False

    ds_count = len(input_files)
    ds_count_ok = 0
    ds_index = 0
    for input_file in input_files:
        monitor(f'processing dataset {ds_index + 1} of {ds_count}: {input_file!r}...')
        # noinspection PyTypeChecker
        output_path, status = _process_l2_input(input_processor,
                                                input_reader,
                                                effective_input_reader_params,
                                                output_writer,
                                                effective_output_writer_params,
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
        ds_index += 1
        if status:
            ds_count_ok += 1

    monitor(f'{ds_count_ok} of {ds_count} datasets processed successfully, '
            f'{ds_count - ds_count_ok} were dropped due to errors')

    return output_path, status


def _process_l2_input(input_processor: InputProcessor,
                      input_reader: DatasetIO,
                      input_reader_params: Dict[str, Any],
                      output_writer: DatasetIO,
                      output_writer_params: Dict[str, Any],
                      input_file: str,
                      output_size: Tuple[int, int],
                      output_region: Tuple[float, float, float, float],
                      output_resampling: str,
                      output_dir: str,
                      output_name: str,
                      output_metadata: NameAnyDict = None,
                      output_variables: NameDictPairList = None,
                      processed_variables: NameDictPairList = None,
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
        input_dataset = input_reader.read(input_file, **input_reader_params)
    except Exception as e:
        monitor(f'ERROR: cannot read input: {e}: skipping...')
        traceback.print_exc()
        return None, False

    reprojection_info = input_processor.get_reprojection_info(input_dataset)
    time_range = input_processor.get_time_range(input_dataset)

    if output_variables:
        output_variables = to_resolved_name_dict_pairs(output_variables, input_dataset)
    else:
        output_variables = [(var_name, None) for var_name in input_dataset.data_vars]

    selected_variables = set([var_name for var_name, _ in output_variables])
    if reprojection_info is not None:
        if reprojection_info.xy_var_names:
            selected_variables.update(reprojection_info.xy_var_names)
        if reprojection_info.xy_tp_var_names:
            selected_variables.update(reprojection_info.xy_tp_var_names)

    steps = []

    # noinspection PyShadowingNames
    def step1(dataset):
        return input_processor.pre_process(dataset)

    steps.append((step1, 'pre-processing dataset'))

    # noinspection PyShadowingNames
    def step2(dataset):
        return compute_dataset(dataset, processed_variables=processed_variables)

    steps.append((step2, 'computing variables'))

    # noinspection PyShadowingNames
    def step3(dataset):
        return select_variables(dataset, selected_variables)

    steps.append((step3, 'selecting variables'))

    if reprojection_info is not None:
        # noinspection PyShadowingNames
        def step4(dataset):
            return reproject_to_wgs84(dataset,
                                      src_xy_var_names=reprojection_info.xy_var_names,
                                      src_xy_tp_var_names=reprojection_info.xy_tp_var_names,
                                      src_xy_crs=reprojection_info.xy_crs,
                                      src_xy_gcp_step=reprojection_info.xy_gcp_step or 1,
                                      src_xy_tp_gcp_step=reprojection_info.xy_tp_gcp_step or 1,
                                      dst_size=output_size,
                                      dst_region=output_region,
                                      dst_resampling=output_resampling,
                                      include_non_spatial_vars=False)

        steps.append((step4, 'reprojecting dataset'))

    if time_range is not None:
        # noinspection PyShadowingNames
        def step5(dataset):
            return add_time_coords(dataset, time_range)

        steps.append((step5, 'adding time coordinates'))

    # noinspection PyShadowingNames
    def step6(dataset):
        return update_variable_props(dataset, output_variables)

    steps.append((step6, 'updating variable properties'))

    # noinspection PyShadowingNames
    def step7(dataset):
        return input_processor.post_process(dataset)

    steps.append((step7, 'post-processing dataset'))

    # noinspection PyShadowingNames
    def step8(dataset):
        return update_global_attributes(dataset, output_metadata=output_metadata)

    steps.append((step8, 'updating dataset attributes'))

    if not dry_run:
        if append_mode and os.path.exists(output_path):
            # noinspection PyShadowingNames
            def step9(dataset):
                output_writer.append(dataset, output_path, **output_writer_params)
                return dataset

            steps.append((step9, f'appending to {output_path}'))
        else:
            # noinspection PyShadowingNames
            def step9(dataset):
                rimraf(output_path)
                output_writer.write(dataset, output_path, **output_writer_params)
                return dataset

            steps.append((step9, f'writing to {output_path}'))

    if _PROFILING_ON:
        pr = cProfile.Profile()
        pr.enable()

    try:
        num_steps = len(steps)
        dataset = input_dataset
        total_t1 = time.clock()
        for step_index in range(num_steps):
            transform, label = steps[step_index]
            step_t1 = time.clock()
            monitor(f'step {step_index + 1} of {num_steps}: {label}...')
            dataset = transform(dataset)
            step_t2 = time.clock()
            monitor(f'  {label} completed in {step_t2 - step_t1} seconds')
        total_t2 = time.clock()
        monitor(f'{num_steps} steps took {total_t2 - total_t1} seconds to complete')
    except RuntimeError as e:
        monitor(f'ERROR: during reprojection to WGS84: {e}: skipping it...')
        traceback.print_exc()
        return output_path, False
    finally:
        input_dataset.close()

    if _PROFILING_ON:
        # noinspection PyUnboundLocalVariable
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()
        print(s.getvalue())

    return output_path, True
