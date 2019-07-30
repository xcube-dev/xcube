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

import cProfile
import glob
import io
import json
import numpy as np
import os
import pstats
import shutil
import tempfile
import time
import traceback
import xarray as xr
from typing import Sequence, Callable, Tuple, Dict, Any

from .defaults import DEFAULT_OUTPUT_SIZE, DEFAULT_OUTPUT_RESAMPLING, DEFAULT_OUTPUT_PATH
from .iproc import InputProcessor, get_input_processor
from ..compute import compute_dataset
from ..select import select_vars
from ..update import update_var_props, update_global_attrs
from ...util.config import NameAnyDict, NameDictPairList, to_resolved_name_dict_pairs
from ...util.dsio import rimraf, DatasetIO, find_dataset_io, guess_dataset_format
from ...util.timecoord import add_time_coords, get_time_in_days_since_1970

_PROFILING_ON = False


def gen_cube(input_paths: Sequence[str] = None, input_processor: str = None, input_processor_params: Dict = None,
             input_reader: str = None, input_reader_params: Dict[str, Any] = None,
             output_region: Tuple[float, float, float, float] = None,
             output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE, output_resampling: str = DEFAULT_OUTPUT_RESAMPLING,
             output_path: str = DEFAULT_OUTPUT_PATH,
             output_writer: str = None, output_writer_params: Dict[str, Any] = None,
             output_metadata: NameAnyDict = None, output_variables: NameDictPairList = None,
             processed_variables: NameDictPairList = None, append_mode: bool = False, dry_run: bool = False,
             monitor: Callable[..., None] = None, no_sort: bool = False) -> bool:
    """
    Generate a data cube from one or more input files.

    :param no_sort:
    :param input_paths: The input paths.
    :param input_processor: Name of a registered input processor (xcube.api.gen.inputprocessor.InputProcessor)
           to be used to transform the inputs
    :param input_processor_params: Parameters to be passed to the input processor.
    :param input_reader: Name of a registered input reader (xcube.api.util.dsio.DatasetIO).
    :param input_reader_params: Parameters passed to the input reader.
    :param output_region: Output region as tuple of floats: (lon_min, lat_min, lon_max, lat_max).
    :param output_size: The spatial dimensions of the output as tuple of ints: (width, height).
    :param output_resampling: The resampling method for the output.
    :param output_path: The output directory.
    :param output_writer: Name of an output writer (xcube.api.util.dsio.DatasetIO) used to write the cube.
    :param output_writer_params: Parameters passed to the output writer.
    :param output_metadata: Extra metadata passed to output cube.
    :param output_variables: Output variables.
    :param processed_variables: Processed variables computed on-the-fly.
    :param append_mode: Whether processed inputs shall be appended to an existing cube.
    :param dry_run: Doesn't write any data. For testing.
    :param monitor: A progress monitor.
    :return: True for success.
    """
    # Force loading of plugins
    __import__('xcube.util.plugin')

    if not input_processor:
        raise ValueError('Missing input_processor')

    input_processor = get_input_processor(input_processor)
    if not input_processor:
        raise ValueError(f'Unknown input_processor {input_processor!r}')

    if input_processor_params:
        try:
            input_processor.configure(**input_processor_params)
        except TypeError as e:
            raise ValueError(f'Invalid input_processor_params {input_processor_params!r}') from e

    input_reader = find_dataset_io(input_reader or input_processor.input_reader)
    if not input_reader:
        raise ValueError(f'Unknown input_reader {input_reader!r}')

    if not output_path:
        raise ValueError('Missing output_path')

    output_writer = output_writer or guess_dataset_format(output_path)
    if not output_writer:
        raise ValueError(f'Failed to guess output_writer from path {output_path}')
    output_writer = find_dataset_io(output_writer, modes={'w', 'a'} if append_mode else {'w'})
    if not output_writer:
        raise ValueError(f'Unknown output_writer {output_writer!r}')

    if monitor is None:
        # noinspection PyUnusedLocal
        def monitor(*args):
            pass

    if no_sort is False:
        input_paths = _sort_by_time(input_paths, input_reader, input_processor, monitor)
    else:
        input_paths = [input_file for f in input_paths for input_file in glob.glob(f, recursive=True)]

    if not dry_run:
        output_dir = os.path.abspath(os.path.dirname(output_path))
        os.makedirs(output_dir, exist_ok=True)

    effective_input_reader_params = dict(input_processor.input_reader_params or {})
    effective_input_reader_params.update(input_reader_params or {})

    effective_output_writer_params = output_writer_params or {}

    status = False

    ds_count = len(input_paths)
    ds_count_ok = 0
    ds_index = 0
    for input_file in input_paths:
        monitor(f'processing dataset {ds_index + 1} of {ds_count}: {input_file!r}...')
        # noinspection PyTypeChecker
        status = _process_input(input_processor,
                                input_reader,
                                effective_input_reader_params,
                                output_writer,
                                effective_output_writer_params,
                                input_file,
                                output_size,
                                output_region,
                                output_resampling,
                                output_path,
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

    return status


def _sort_by_time(input_paths: Sequence[str], input_reader, input_processor, monitor):
    """ Sort input paths based on time stamp of each input file"""
    sorted_input_paths = []
    times = []
    input_paths = [input_file for f in input_paths for input_file in glob.glob(f, recursive=True)]
    for input_path in input_paths:
        try:
            input_dataset = input_reader.read(input_path)
            monitor(f'Dataset read for sorting by time:\n{input_path}')
        except Exception as e:
            monitor(f'ERROR: cannot read input: {e}: skipping...')
            traceback.print_exc()
            return False
        time_range = input_processor.get_time_range(input_dataset)
        if _get_half_time(time_range) not in times:
            sorted_input_paths.append(input_path)
            times.append(_get_half_time(time_range))

    times, sorted_input_paths = (list(t) for t in zip(*sorted(zip(times, sorted_input_paths))))

    return sorted_input_paths


def _get_half_time(time_range: Tuple[float, float]):
    """Get half time of time range"""
    start_time = time_range[0]
    end_time = time_range[0]
    half_seconds = (end_time - start_time) / 2
    halftime = start_time + half_seconds
    return halftime


def _process_input(input_processor: InputProcessor,
                   input_reader: DatasetIO,
                   input_reader_params: Dict[str, Any],
                   output_writer: DatasetIO,
                   output_writer_params: Dict[str, Any],
                   input_file: str,
                   output_size: Tuple[int, int],
                   output_region: Tuple[float, float, float, float],
                   output_resampling: str,
                   output_path: str,
                   output_metadata: NameAnyDict = None,
                   output_variables: NameDictPairList = None,
                   processed_variables: NameDictPairList = None,
                   append_mode: bool = False,
                   dry_run: bool = False,
                   monitor: Callable[..., None] = None) -> bool:
    monitor('reading dataset...')
    # noinspection PyBroadException
    try:
        input_dataset = input_reader.read(input_file, **input_reader_params)
        monitor(f'Dataset read:\n{input_dataset}')
    except Exception as e:
        monitor(f'ERROR: cannot read input: {e}: skipping...')
        traceback.print_exc()
        return False

    time_range = input_processor.get_time_range(input_dataset)
    if time_range[0] > time_range[1]:
        monitor('ERROR: start time is greater than end time: skipping...')
        return False

    if output_variables:
        output_variables = to_resolved_name_dict_pairs(output_variables, input_dataset)
    else:
        output_variables = [(var_name, None) for var_name in input_dataset.data_vars]

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
        extra_vars = input_processor.get_extra_vars(dataset)
        selected_variables = set([var_name for var_name, _ in output_variables])
        selected_variables.update(extra_vars or set())
        return select_vars(dataset, selected_variables)

    steps.append((step3, 'selecting variables'))

    # noinspection PyShadowingNames
    def step4(dataset):
        return input_processor.process(dataset,
                                       dst_size=output_size,
                                       dst_region=output_region,
                                       dst_resampling=output_resampling,
                                       include_non_spatial_vars=False)

    steps.append((step4, 'transforming dataset'))

    if time_range is not None:
        # noinspection PyShadowingNames
        def step5(dataset):
            return add_time_coords(dataset, time_range)

        steps.append((step5, 'adding time coordinates'))

    # noinspection PyShadowingNames
    def step6(dataset):
        return update_var_props(dataset, output_variables)

    steps.append((step6, 'updating variable properties'))

    # noinspection PyShadowingNames
    def step7(dataset):
        return input_processor.post_process(dataset)

    steps.append((step7, 'post-processing dataset'))

    # noinspection PyShadowingNames
    def step8(dataset):
        return update_global_attrs(dataset, output_metadata=output_metadata)

    steps.append((step8, 'updating dataset attributes'))

    if not dry_run:
        if append_mode and os.path.exists(output_path):
            # noinspection PyShadowingNames
            if output_path.endswith('.nc'):
                def step9(dataset):
                    output_writer.append(dataset, output_path, **output_writer_params)
                    return dataset

                steps.append((step9, f'appending to {output_path}'))

            elif output_path.endswith('.zarr'):
                append_input_file_to_output_path = _check_append_or_insert(time_range, output_path)
                if append_input_file_to_output_path:
                    def step9(dataset):
                        output_writer.append(dataset, output_path, **output_writer_params)
                        return dataset

                    steps.append((step9, f'appending to {output_path}'))

                else:
                    def step9(dataset):
                        input_tempdir = tempfile.TemporaryDirectory()
                        output_writer.write(dataset, input_tempdir.name, **output_writer_params)
                        merged_data_set = _insert_input_file_into_output_path(input_tempdir.name, output_path)
                        if not merged_data_set:
                            monitor('Time stamp of input data set is already exists in output: skipping it...')
                            return False
                        return dataset

                    steps.append((step9, f'inserting into {output_path}'))

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
        total_t1 = time.perf_counter()
        for step_index in range(num_steps):
            transform, label = steps[step_index]
            step_t1 = time.perf_counter()
            monitor(f'step {step_index + 1} of {num_steps}: {label}...')
            dataset = transform(dataset)
            step_t2 = time.perf_counter()
            monitor(f'  {label} completed in {step_t2 - step_t1} seconds')
        total_t2 = time.perf_counter()
        monitor(f'{num_steps} steps took {total_t2 - total_t1} seconds to complete')
    except RuntimeError as e:
        monitor(f'ERROR: during reprojection to WGS84: {e}: skipping it...')
        traceback.print_exc()
        return False
    finally:
        input_dataset.close()

    if _PROFILING_ON:
        # noinspection PyUnboundLocalVariable
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()
        print(s.getvalue())

    return True


def _check_append_or_insert(time_range: Tuple[float, float], output_path: str) -> bool:
    """Checking if time stamp needs to be inserted into output path or appended."""
    t1, t2 = time_range
    if t1 != t2:
        t_center = (t1 + t2) / 2
    else:
        t_center = t1
    with xr.open_zarr(output_path) as ds:
        return np.greater(t_center, get_time_in_days_since_1970(ds.time.values[-1]))


def _insert_input_file_into_output_path(input_path: str, output_path: str) -> bool:
    """Merging the data for the new time stamp into the existing and remaining zarr directory."""
    insert = _check_if_insert_into_output_path(input_path, output_path)
    if insert:
        with xr.open_zarr(input_path) as input_ds, xr.open_zarr(output_path) as output_ds:
            input_time_idx = 0
            new_time_idx = np.amin(np.where(output_ds.time[:] > (input_ds.time.values[input_time_idx])))
            # Preparing the source directory with the single time stamp to be ready for merging
            # --> files of variables, excluding "lat" and "lon" need to be renamed
            _rename_file(input_path, input_time_idx, new_time_idx)
            # Preparing the destination directory to be ready for single time stam to be merged
            # --> files of variables, excluding "lat" and "lon" need to be renamed
            # The renaming needs to happen in reversed order and starting at the index of nearest above value:
            for i in reversed(range(new_time_idx, output_ds.time.shape[0])):
                _rename_file(output_path, i, (i + 1))
            # Final step: copy the single time stamp files into destination zarr and adjusting .zarray to the change.
            _copy_into_output_path(input_path, output_path, new_time_idx)
            return True
    else:
        return False


def _check_if_insert_into_output_path(input_path: str, output_path: str) -> bool:
    """Check if to be added time stamp is unique """
    with xr.open_zarr(input_path) as input_ds, xr.open_zarr(output_path) as output_ds:
        mask = np.equal(output_ds.time.values, input_ds.time.values)
        return not mask.any()


def _rename_file(path: str, old_time_idx: int, new_time_idx: int):
    """Renaming files within the directories according to new time index."""
    for variable in _get_variables_and_dims_to_rename_and_copy(path):
        v_path = os.path.join(path, variable)
        for root, dirs, files in os.walk(v_path):
            for filename in files:
                parts1 = filename.split('.', 1)[0]
                if parts1 == (str(old_time_idx)) and (variable != "time"):
                    parts2 = filename.split('.', 1)[1]
                    new_name = f'{new_time_idx}.{parts2}'
                    os.rename(os.path.join(v_path, filename), os.path.join(v_path, new_name))
                elif parts1 == (str(old_time_idx)) and (variable == "time"):
                    os.rename(os.path.join(v_path, filename), os.path.join(v_path, str(new_time_idx)))


def _copy_into_output_path(input_path: str, output_path: str, input_time_idx: int):
    """Copy the files with the new time stamp into the existing zarr directory."""
    for variable in _get_variables_and_dims_to_rename_and_copy(input_path):
        v_path = os.path.join(input_path, variable)
        for root, dirs, files in os.walk(v_path):
            for filename in files:
                parts1 = filename.split('.', 1)[0]
                if parts1 == str(input_time_idx):
                    shutil.copyfile((os.path.join(input_path, variable, filename)),
                                    (os.path.join(output_path, variable, filename)))
        _adjust_zarray(output_path, variable)


def _get_variables_and_dims_to_rename_and_copy(path: str):
    with xr.open_zarr(path) as ds:
        variables = []
        for v in ds.variables:
            if 'time' in ds[v].dims:
                variables.append(v)
    return variables


def _adjust_zarray(output_path: str, variable: str):
    """Changing the shape for time in the .zarray file."""
    with open((os.path.join(output_path, variable, '.zarray')), 'r') as jsonFile:
        data = json.load(jsonFile)
    t_shape = data["shape"]
    data["shape"][0] = t_shape[0] + 1

    with open((os.path.join(output_path, variable, '.zarray')), 'w') as jsonFile:
        json.dump(data, jsonFile, indent=4)
