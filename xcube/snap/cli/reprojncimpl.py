import glob
import os
from abc import abstractmethod
from typing import Sequence, Callable, Tuple, Set

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
        old_ds = xr.open_dataset(temp_path)
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


def reproj_nc(input_files: Sequence[str],
              dst_size: Tuple[int, int],
              dst_region: Tuple[float, float, float, float],
              dst_variables: Set[str],
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

    input_files = [input_file for f in input_files for input_file in glob.glob(f, recursive=True)]

    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        monitor('reading %s...' % input_file)
        dataset = xr.open_dataset(input_file, decode_cf=True, decode_coords=True)

        if dst_variables:
            dropped_variables = set(dataset.data_vars.keys()).difference(dst_variables)
            if dropped_variables:
                dataset = dataset.drop(dropped_variables)

        monitor('masking...')
        masked_dataset, mask_sets = mask_dataset(dataset,
                                                 expr_pattern='({expr}) AND !quality_flags.land',
                                                 errors='raise')

        for _, mask_set in mask_sets.items():
            monitor('mask set found: %s' % mask_set)

        proj_dataset = reproject_to_wgs84(masked_dataset,
                                          dst_size,
                                          dst_region=dst_region,
                                          gcp_i_step=50)

        basename = os.path.basename(input_file)
        basename, ext = basename.rsplit('.', 1) if '.' in basename else (basename, None)

        output_name = output_name.format(INPUT_FILE=basename)
        output_basename = output_name + '.' + dataset_writer.ext
        output_path = os.path.join(output_dir, output_basename)

        if append and os.path.exists(output_path):
            monitor('appending to %s...' % output_path)
            dataset_writer.append(proj_dataset, output_path)
        else:
            _rm(output_path)
            monitor('writing %s...' % output_path)
            dataset_writer.create(proj_dataset, output_path)

        proj_dataset.close()

        # from matplotlib import pyplot as plt
        # for var_name in new_dataset.variables:
        #     var = new_dataset[var_name]
        #     if var.dims == ('lat', 'lon'):
        #         var.plot()
        #         plt.show()


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
