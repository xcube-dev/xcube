import glob
import os
from abc import abstractmethod
from typing import Sequence, Callable, Tuple, Set, Any, Dict, Optional

import xarray as xr
import zarr
from xcube.reproject import reproject_to_wgs84
from xcube.snap.mask import mask_dataset


class DatasetWriter:
    @property
    @abstractmethod
    def ext(self) -> str:
        pass

    @abstractmethod
    def create(self, dataset: xr.Dataset, output_path: str):
        pass

    @abstractmethod
    def append(self, dataset: xr.Dataset, output_path: str):
        pass


class Netcdf4Writer(DatasetWriter):

    @property
    def ext(self) -> str:
        return 'nc'

    def create(self, dataset: xr.Dataset, output_path: str):
        dataset.to_netcdf(output_path)

    def append(self, dataset: xr.Dataset, output_path: str):
        import os
        temp_path = output_path + 'temp.nc'
        os.rename(output_path, temp_path)
        old_ds = xr.open_dataset(temp_path, decode_times=False)
        new_ds = xr.concat([old_ds, dataset],
                           dim='time',
                           data_vars='minimal',
                           coords='minimal',
                           compat='equals')
        new_ds.to_netcdf(output_path)
        old_ds.close()
        _rm(temp_path)


class ZarrWriter(DatasetWriter):
    def __init__(self):
        self.root_group = None

    @property
    def ext(self) -> str:
        return 'zarr'

    def create(self, dataset: xr.Dataset, output_path: str):
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        encoding = dict()
        for var_name in dataset.data_vars:
            new_var = dataset[var_name]
            encoding[var_name] = {'compressor': compressor, 'chunks': new_var.shape}
        dataset.to_zarr(output_path,
                        encoding=encoding)

    def append(self, dataset: xr.Dataset, output_path: str):
        import zarr
        if self.root_group is None:
            self.root_group = zarr.open(output_path, mode='a')
        for var_name, var_array in self.root_group.arrays():
            new_var = dataset[var_name]
            if 'time' in new_var.dims:
                axis = new_var.dims.index('time')
                var_array.append(new_var, axis=axis)


def reproj_nc_files(input_files: Sequence[str],
                    dst_size: Tuple[int, int],
                    dst_region: Tuple[float, float, float, float],
                    dst_variables: Set[str],
                    dst_metadata: Optional[Dict[str, Any]],
                    output_dir: str,
                    output_name: str,
                    output_format: str,
                    append: bool,
                    monitor: Callable[..., None] = None):
    if output_format == 'nc' or output_format == 'netcdf4':
        dataset_writer = Netcdf4Writer()
    elif output_format == 'zarr':
        dataset_writer = ZarrWriter()
    else:
        raise ValueError(f'unknown output format {output_format!r}')

    if monitor is None:
        # noinspection PyUnusedLocal
        def monitor(*args):
            pass

    input_files = sorted([input_file for f in input_files for input_file in glob.glob(f, recursive=True)])

    os.makedirs(output_dir, exist_ok=True)

    ds_count = len(input_files)
    ds_index = 0
    output_path = None
    for input_file in input_files:
        monitor(f'processing dataset {ds_index + 1} of {ds_count}: {input_file!r}...')
        output_path, ok = reproj_nc_file(input_file,
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


def reproj_nc_file(input_file: str,
                   dst_size: Tuple[int, int],
                   dst_region: Tuple[float, float, float, float],
                   dst_variables: Set[str],
                   dst_metadata: Dict[str, Any],
                   output_dir: str,
                   output_name: str,
                   dataset_writer: DatasetWriter,
                   append: bool,
                   monitor: Callable[..., None] = None):
    basename = os.path.basename(input_file)
    basename, ext = basename.rsplit('.', 1) if '.' in basename else (basename, None)

    output_name = output_name.format(INPUT_FILE=basename)
    output_basename = output_name + '.' + dataset_writer.ext
    output_path = os.path.join(output_dir, output_basename)

    monitor('reading...')
    dataset = xr.open_dataset(input_file, decode_cf=True, decode_coords=True, decode_times=False)

    if dst_variables:
        dropped_variables = set(dataset.data_vars.keys()).difference(dst_variables)
        if dropped_variables:
            dataset = dataset.drop(dropped_variables)

    monitor('masking...')
    masked_dataset, mask_sets = mask_dataset(dataset,
                                             expr_pattern='({expr}) AND !quality_flags.land',
                                             errors='raise')

    try:
        proj_dataset = reproject_to_wgs84(masked_dataset,
                                          dst_size,
                                          dst_region=dst_region,
                                          gcp_i_step=5)
    except RuntimeError as e:
        import sys
        import traceback
        monitor(f'ERROR: during reprojection to WGS84: {e}')
        monitor('skipping dataset')
        etype, value, tb = sys.exc_info()
        traceback.print_exception(etype, value, tb)
        return output_path, False

    if dst_metadata:
        proj_dataset.attrs.clear()
        proj_dataset.attrs.update(dst_metadata)

    if append and os.path.exists(output_path):
        monitor(f'appending to {output_path}...')
        dataset_writer.append(proj_dataset, output_path)
    else:
        _rm(output_path)
        monitor(f'writing to {output_path}...')
        dataset_writer.create(proj_dataset, output_path)

    proj_dataset.close()

    return output_path, True


def _rm(path):
    import os
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except:
            pass
