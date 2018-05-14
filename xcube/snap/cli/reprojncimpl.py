import glob
import os
from typing import Sequence, Callable, Tuple, Set

import xarray as xr
import zarr

from xcube.reproject import reproject_to_wgs84
from xcube.snap.mask import mask_dataset


def reproj_nc(input_files: Sequence[str],
              dst_size: Tuple[int, int],
              dst_region: Tuple[float, float, float, float],
              dst_variables: Set[str],
              output_dir: str,
              output_name: str,
              output_format: str,
              append: bool,
              monitor: Callable[..., None] = None):
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
        output_basename = output_name + '.' + output_format
        output_path = os.path.join(output_dir, output_basename)

        _rm(output_path)

        if append and os.path.exists(output_path):
            monitor('appending to %s...' % output_path)
            old_ds = xr.open_dataset(output_path)
            proj_dataset = xr.concat([old_ds, proj_dataset],
                                     dim='time',
                                     data_vars='minimal',
                                     coords='minimal',
                                     compat='equals')
            old_ds.close()

        monitor('writing %s...' % output_path)

        if output_format == 'nc':
            proj_dataset.to_netcdf(output_path)
        elif output_format == 'zarr':
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
            proj_dataset.to_zarr(output_path,
                                 encoding={output_name: {'compressor': compressor}})

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
