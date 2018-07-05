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

import argparse
import sys
import traceback
from typing import List, Optional

from xcube.genl3.process import generate_l3_cube, DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_PATTERN, \
    DEFAULT_OUTPUT_RESAMPLING_METHOD, \
    DEFAULT_OUTPUT_FORMAT, OUTPUT_FORMAT_NAMES, RESAMPLING_METHODS, DEFAULT_OUTPUT_FREQUENCY
from xcube.metadata import load_metadata_yaml
from xcube.version import version

__import__('xcube.plugin')


def main(args: Optional[List[str]] = None):
    """
    Generate L2C data cubes from L2 data products.
    """

    parser = argparse.ArgumentParser(description='Generate Level-3 data cube from Level-2C data cube.')
    parser.add_argument('--version', '-V', action='version', version=version)
    parser.add_argument('--dir', '-d', dest='output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory. Defaults to {DEFAULT_OUTPUT_DIR!r}')
    parser.add_argument('--name', '-n', dest='output_name', default=DEFAULT_OUTPUT_PATTERN,
                        help=f'Output filename pattern. Defaults to {DEFAULT_OUTPUT_PATTERN!r}.')
    parser.add_argument('--format', '-f', dest='output_format',
                        default=DEFAULT_OUTPUT_FORMAT, choices=OUTPUT_FORMAT_NAMES,
                        help=f'Output format. Defaults to {DEFAULT_OUTPUT_FORMAT!r}.')
    parser.add_argument('--variables', '-v', dest='output_variables',
                        help='Variables to be included in output. '
                             'Comma-separated list of names which may contain wildcard characters "*" and "?".')
    parser.add_argument('--resampling', dest='output_resampling', choices=RESAMPLING_METHODS,
                        help='Temporal resampling method. Use format "<count><offset>"'
                             'where <offset> is one of {H, D, W, M, Q, Y}'
                             f'Defaults to {DEFAULT_OUTPUT_RESAMPLING_METHOD!r}.')
    parser.add_argument('--frequency', dest='output_frequency',
                        help='Temporal aggregation frequency.'
                             f'Defaults to {DEFAULT_OUTPUT_FREQUENCY!r}.')
    parser.add_argument('--meta-file', '-m', dest='output_meta_file',
                        help='File containing cube-level, CF-compliant metadata in YAML format.')
    parser.add_argument('--dry-run', default=False, action='store_true',
                        help='Just read and process inputs, but don\'t produce any outputs.')
    parser.add_argument('--traceback', dest='print_traceback', default=False, action='store_true',
                        help='On error, print Python traceback.')
    parser.add_argument('input_file', metavar='INPUT_FILE',
                        help="The input file or directory which must be a Level-2C cube.")

    try:
        arg_obj = parser.parse_args(args or sys.argv[1:])
    except SystemExit as e:
        return int(str(e))

    input_file = arg_obj.input_file
    output_dir = arg_obj.output_dir or DEFAULT_OUTPUT_DIR
    output_name = arg_obj.output_name or DEFAULT_OUTPUT_PATTERN
    output_format = arg_obj.output_format or DEFAULT_OUTPUT_FORMAT
    output_variables = arg_obj.output_variables
    output_resampling = arg_obj.output_resampling or DEFAULT_OUTPUT_RESAMPLING_METHOD
    output_frequency = arg_obj.output_frequency or DEFAULT_OUTPUT_FREQUENCY
    output_meta_file = arg_obj.output_meta_file
    print_traceback = arg_obj.print_traceback
    dry_run = arg_obj.dry_run

    if output_variables:
        try:
            output_variables = set(map(lambda c: str(c).strip(), output_variables.split(',')))
        except ValueError:
            output_variables = None
        if output_variables is not None \
                and next(iter(True for var_name in output_variables if var_name == ''), False):
            output_variables = None
        if output_variables is None or len(output_variables) == 0:
            print(f'error: invalid variables {arg_obj.output_variables!r}')
            return 2

    if output_meta_file:
        try:
            with open(output_meta_file) as stream:
                output_metadata = load_metadata_yaml(stream)
            print(f'loaded metadata from file {arg_obj.output_meta_file!r}')
        except OSError as e:
            print(f'error: failed loading metadata file {arg_obj.output_meta_file!r}: {e}')
            if print_traceback:
                traceback.print_exc()
            return 2
    else:
        output_metadata = None

    try:
        generate_l3_cube(input_file,
                         output_variables,
                         output_metadata,
                         output_dir,
                         output_name,
                         output_format,
                         output_resampling,
                         output_frequency,
                         dry_run=dry_run,
                         monitor=print)
    except BaseException as e:
        print(f'error: {e}')
        if print_traceback:
            traceback.print_exc()
        return 2

    return 0


if __name__ == '__main__':
    sys.exit(status=main())
