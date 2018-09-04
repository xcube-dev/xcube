import os
import random
import subprocess
import sys

import xarray as xr
import zarr

# If using Blosc in a multi-process program then it is recommended to disable multi-threading
zarr.blosc.use_threads = False


def main(args=None):
    args = args if args is not None else sys.argv[1:]
    if len(args) != 2:
        print(f'Usage: {sys.argv[0]} OUTPUT.zarr (INPUT.nc | INPUT.dir)')
        sys.exit(2)

    output_dir = args[0]
    input_file = args[1]

    if os.path.isdir(input_file):
        input_dir = input_file
        input_files = list(os.listdir(input_dir))
        # Shuffle files
        for i in range(len(input_files)):
            i1 = random.randint(0, len(input_files) - 1)
            i2 = random.randint(0, len(input_files) - 1)
            t = input_files[i1]
            input_files[i1] = input_files[i2]
            input_files[i2] = t
        for input_file in input_files:
            print(f'processing {input_file}')
            subprocess.run([sys.executable, sys.argv[0], output_dir, os.path.join(input_dir, input_file)])
        return

    synchronizer = zarr.ProcessSynchronizer(output_dir + '.sync')
    input_ds = xr.open_dataset(input_file, decode_times=False)
    dropped_vars = set(input_ds.data_vars.keys()) - {"analysed_sst", "analysis_error"}
    input_ds = input_ds.drop(dropped_vars)

    if not os.path.isdir(output_dir):
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        encoding = dict()
        for var_name in input_ds.data_vars:
            new_var = input_ds[var_name]
            chunks = new_var.shape
            encoding[var_name] = {'compressor': compressor, 'chunks': chunks}
        input_ds.to_zarr(output_dir, encoding=encoding, synchronizer=synchronizer)
        print(f'written {input_file} to {output_dir}')
    else:
        # cube_ds = xr.open_zarr(output_dir, synchronizer=synchronizer)
        # cube_ds = xr.concat([cube_ds, input_ds], dim='time')
        # cube_ds.close()
        root_group = zarr.open(output_dir, mode='a', synchronizer=synchronizer)
        for var_name, var_array in root_group.arrays():
            if var_name in input_ds:
                var = input_ds[var_name]
                if 'time' in var.dims:
                    if var_name == 'time':
                        print('time:', var, var.values)
                    axis = var.dims.index('time')
                    # Note: all append operations are forced to be sequential!
                    # See https://github.com/zarr-developers/zarr/issues/75
                    var_array.append(var, axis=axis)
        print(f'appended {input_file} to {output_dir}')


if __name__ == "__main__":
    main()
