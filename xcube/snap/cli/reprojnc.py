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
                        help='Output directory.')
    parser.add_argument('--name', '-n', dest='output_name', default=DEFAULT_OUTPUT_PATTERN,
                        help='Output filename pattern.')
    parser.add_argument('--format', '-f', dest='output_format', default=DEFAULT_OUTPUT_FORMAT, choices=['nc', 'zarr'],
                        help='Output format')
    parser.add_argument('--size', '-s', dest='dst_size', default=DEFAULT_DST_SIZE,
                        help='Output size in pixels using format "<width>,<height>".')
    parser.add_argument('--region', '-r', dest='dst_region',
                        help='Output region using format "<lon-min>,<lat-min>,<lon-max>,<lat-max>"')
    parser.add_argument('--variables', '-v', dest='dst_variables',
                        help='Variables to be included in output. Comma-separated list of names.')
    parser.add_argument('--append', '-a', default=False, action='store_true',
                        help='Append successive outputs.')
    parser.add_argument('input_files', metavar='INPUT_FILES', nargs='+',
                        help="SNAP NetCDF4 products. May contain wildcards.")

    arg_obj = parser.parse_args(args)

    input_files = arg_obj.input_files
    output_dir = arg_obj.output_dir or DEFAULT_OUTPUT_DIR
    output_name = arg_obj.output_name or DEFAULT_OUTPUT_PATTERN
    output_format = arg_obj.output_format or DEFAULT_OUTPUT_FORMAT
    dst_size = arg_obj.dst_size
    dst_region = arg_obj.dst_region
    dst_variables = arg_obj.dst_variables
    append = arg_obj.append

    if dst_size:
        dst_size = list(map(lambda c: int(c), dst_size.split(',')))
        if len(dst_size) != 2:
            print(f'error: invalid size {arg_obj.dst_size!r}')
            sys.exit(10)

    if dst_region:
        dst_region = list(map(lambda c: float(c), dst_region.split(',')))
        if len(dst_region) != 4:
            print(f'error: invalid region {arg_obj.dst_region!r}')
            sys.exit(10)

    if dst_variables:
        dst_variables = set(map(lambda c: str(c).strip(), dst_variables.split(',')))
        if len(dst_variables) == 0:
            print(f'error: invalid variables {arg_obj.dst_variables!r}')
            sys.exit(10)

    from xcube.snap.cli.reprojncimpl import reproj_nc_files
    reproj_nc_files(input_files,
                    dst_size,
                    dst_region,
                    dst_variables,
                    output_dir,
                    output_name,
                    output_format,
                    append,
                    monitor=print)


if __name__ == '__main__':
    main()
