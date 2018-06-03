import glob
import os
from typing import Sequence, Callable, Tuple, Set, Any, Dict, Optional

import xarray as xr

from .inputprocessor import InputProcessor
from ..io import rimraf, get_default_dataset_io_registry, DatasetIO
from ..reproject import reproject_to_wgs84


def process_inputs(input_files: Sequence[str],
                   input_type: str,
                   dst_size: Tuple[int, int],
                   dst_region: Tuple[float, float, float, float],
                   dst_variables: Set[str],
                   dst_metadata: Optional[Dict[str, Any]],
                   output_dir: str,
                   output_name: str,
                   output_format: str,
                   append: bool,
                   monitor: Callable[..., None] = None):
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

    os.makedirs(output_dir, exist_ok=True)

    ds_count = len(input_files)
    ds_index = 0
    for input_file in input_files:
        monitor(f'processing dataset {ds_index + 1} of {ds_count}: {input_file!r}...')
        # noinspection PyTypeChecker
        output_path, ok = process_input(input_file,
                                        input_processor,
                                        dst_size,
                                        dst_region,
                                        dst_variables,
                                        dst_metadata,
                                        output_dir,
                                        output_name,
                                        dataset_writer,
                                        append,
                                        monitor)
        if ok:
            ds_index += 1

    # if dst_metadata and append and output_path:
    #     monitor(f'adding file-level metadata to {output_path!r}...')
    #     if output_format == 'nc':
    #         ds = xr.open_dataset(output_path)
    #         ds.attrs.clear()
    #         ds.attrs.update(dst_metadata)
    #         ds.to_netcdf(output_path)
    #         ds.close()
    #     elif output_format == 'zarr':
    #         ds = xr.open_zarr(output_path)
    #         ds.attrs.clear()
    #         ds.attrs.update(dst_metadata)
    #         ds.to_zarr(output_path)
    #         ds.close()
    #     monitor(f'done adding file-level metadata.')

    monitor(f'{ds_index} of {ds_count} datasets processed successfully, '
            f'{ds_count - ds_index} were dropped due to errors')


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
                  monitor: Callable[..., None] = None):
    basename = os.path.basename(input_file)
    basename, ext = basename.rsplit('.', 1) if '.' in basename else (basename, None)

    output_name = output_name.format(INPUT_FILE=basename)
    output_basename = output_name + '.' + dataset_writer.ext
    output_path = os.path.join(output_dir, output_basename)

    monitor('reading...')
    dataset = input_processor.read(input_file)

    if dst_variables:
        dropped_variables = set(dataset.data_vars.keys()).difference(dst_variables)
        if dropped_variables:
            dataset = dataset.drop(dropped_variables)


    try:
        monitor('pre-processing...')
        dataset = input_processor.pre_reproject(dataset)
        monitor('reprojecting...')
        dataset = reproject_to_wgs84(dataset,
                                          dst_size,
                                          dst_region=dst_region,
                                          gcp_i_step=5)
        monitor('pos-processing...')
        dataset = input_processor.post_reproject(dataset)
    except RuntimeError as e:
        import sys
        import traceback
        monitor(f'ERROR: during reprojection to WGS84: {e}')
        monitor('skipping dataset')
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        return output_path, False

    if dst_metadata:
        dataset.attrs.clear()
        dataset.attrs.update(dst_metadata)

    if append and os.path.exists(output_path):
        monitor(f'appending to {output_path}...')
        dataset_writer.append(dataset, output_path)
    else:
        rimraf(output_path)
        monitor(f'writing to {output_path}...')
        dataset_writer.write(dataset, output_path)

    dataset.close()

    return output_path, True
