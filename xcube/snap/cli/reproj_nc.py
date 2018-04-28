import argparse
import os
import sys
from typing import List, Optional

import xarray as xr
import zarr

from xcube.reproject import reproject_to_wgs84
from xcube.snap.mask import mask_dataset
from xcube.version import __version__ as version

DEFAULT_OUTPUT_DIR = '.'
DEFAULT_OUTPUT_PATTERN = 'PROJ_WGS84_{INPUT_FILE}'
DEFAULT_OUTPUT_FORMAT = 'nc'
DEFAULT_DST_SIZE = '512,512'


def main(args: Optional[List[str]] = None):
    """
    Masks and reprojects a SNAP OLCI L2 NetCDF4 product.

    Note, this is not yet a tool rather than an end-to-end test.

    Tested with HIGHROC/0001_SNS/OLCI/2017/04/O_L2_0001_SNS_2017104102450_v1.0.nc
    """
    args = args or sys.argv[1:]

    parser = argparse.ArgumentParser(description='Reproject SNAP NetCDF4 product')
    parser.add_argument('--version', '-V', action='version', version=version)
    parser.add_argument('--dir', '-d', dest='output_dir', default=DEFAULT_OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--pattern', '-p', dest='output_pattern', default=DEFAULT_OUTPUT_PATTERN,
                        help='Output filename pattern')
    parser.add_argument('--format', '-f', dest='output_format', default=DEFAULT_OUTPUT_FORMAT, choices=['nc', 'zarr'],
                        help='Output format')
    parser.add_argument('--size', '-s', dest='dst_size', default=DEFAULT_DST_SIZE,
                        help='Output size in pixels using format "<width>,<height>"')
    parser.add_argument('--region', '-r', dest='dst_region',
                        help='Output region using format "<lon-min>,<lat-min>,<lon-max>,<lat-max>"')
    parser.add_argument('input_file',
                        help="SNAP NetCDF4 product")

    arg_obj = parser.parse_args(args)

    input_file = arg_obj.input_file
    output_dir = arg_obj.output_dir or DEFAULT_OUTPUT_DIR
    output_pattern = arg_obj.output_pattern or DEFAULT_OUTPUT_PATTERN
    output_format = arg_obj.output_format or DEFAULT_OUTPUT_FORMAT
    dst_size = arg_obj.dst_size
    dst_region = arg_obj.dst_region

    if dst_size:
        dst_size = dst_size.split(',').map(lambda c: int(c))
        if len(dst_region) != 2:
            print('error: invalid size "%s"' % arg_obj.dst_size)
            sys.exit(10)

    if dst_region:
        dst_region = dst_region.split(',').map(lambda c: float(c))
        if len(dst_region) != 4:
            print('error: invalid region "%s"' % arg_obj.dst_region)
            sys.exit(10)

    print('reading %s...' % input_file)
    dataset = xr.open_dataset(input_file, decode_cf=True, decode_coords=True)

    print('masking...')
    masked_dataset, mask_sets = mask_dataset(dataset,
                                             expr_pattern='({expr}) AND !quality_flags.land',
                                             errors='raise')

    for _, mask_set in mask_sets.items():
        print('mask set found: %s' % mask_set)

    proj_dataset = reproject_to_wgs84(masked_dataset,
                                      dst_size,
                                      dst_region=dst_region,
                                      gcp_i_step=50)

    basename = os.path.basename(input_file)
    basename, ext = basename.rsplit('.', 1) if '.' in basename else (basename, None)

    output_name = output_pattern.format(INPUT=basename)
    output_basename = output_name + '.' + output_format
    output_path = os.path.join(output_dir, output_basename)

    _rm(output_path)

    print('writing %s...' % output_path)

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


if __name__ == '__main__':
    main()
