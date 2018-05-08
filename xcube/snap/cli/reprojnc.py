import argparse
import sys
from typing import List, Optional

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
    parser.add_argument('--append', '-a', default=False, action='store_true',
                        help='Output size in pixels using format "<width>,<height>"')
    parser.add_argument('input_files', metavar='INPUT_FILES', nargs='+',
                        help="SNAP NetCDF4 products. May contain wildcards.")

    arg_obj = parser.parse_args(args)

    input_files = arg_obj.input_files
    output_dir = arg_obj.output_dir or DEFAULT_OUTPUT_DIR
    output_pattern = arg_obj.output_pattern or DEFAULT_OUTPUT_PATTERN
    output_format = arg_obj.output_format or DEFAULT_OUTPUT_FORMAT
    dst_size = arg_obj.dst_size
    dst_region = arg_obj.dst_region

    if dst_size:
        dst_size = list(map(lambda c: int(c), dst_size.split(',')))
        if len(dst_size) != 2:
            print('error: invalid size "%s"' % arg_obj.dst_size)
            sys.exit(10)

    if dst_region:
        dst_region = list(map(lambda c: float(c), dst_region.split(',')))
        if len(dst_region) != 4:
            print('error: invalid region "%s"' % arg_obj.dst_region)
            sys.exit(10)

    from .reprojncimpl import reproj_nc
    reproj_nc(input_files, dst_size, dst_region, output_pattern, output_format, output_dir, monitor=print)


if __name__ == '__main__':
    main()
