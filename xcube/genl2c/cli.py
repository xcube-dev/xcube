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
from typing import List, Optional

from xcube.genl2c.inputprocessor import InputProcessor
# from xcube.genl2c.snap.inputprocessor import SnapOlciHighrocL2InputProcessor
from xcube.io import get_default_dataset_io_registry
from xcube.metadata import load_yaml
from xcube.version import version

__import__('xcube.plugin')

DEFAULT_OUTPUT_DIR = '.'
DEFAULT_OUTPUT_PATTERN = 'PROJ_WGS84_{INPUT_FILE}'
DEFAULT_OUTPUT_FORMAT = 'nc'
DEFAULT_output_size = '512,512'


def main(args: Optional[List[str]] = None):
    """
    Generate L2C data cubes from L2 data products.
    """
    ds_io_registry = get_default_dataset_io_registry()
    input_ds_ios = ds_io_registry.query(lambda ds_io: isinstance(ds_io, InputProcessor))
    output_ds_ios = ds_io_registry.query(lambda ds_io: 'w' in ds_io.modes)
    input_type_names = [ds_io.name for ds_io in input_ds_ios]
    output_format_names = [ds_io.name for ds_io in output_ds_ios]

    parser = argparse.ArgumentParser(description='Generate L2C data cube from various input files. '
                                                 'L2C data cubes may be created in one go or in successively '
                                                 'in append mode, input by input.',
                                     formatter_class=GenL2CHelpFormatter)
    parser.add_argument('--version', '-V', action='version', version=version)
    parser.add_argument('--dir', '-d', dest='output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory. Defaults to {DEFAULT_OUTPUT_DIR!r}')
    parser.add_argument('--name', '-n', dest='output_name', default=DEFAULT_OUTPUT_PATTERN,
                        help=f'Output filename pattern. Defaults to {DEFAULT_OUTPUT_PATTERN!r}.')
    parser.add_argument('--format', '-f', dest='output_format',
                        default=output_format_names[0], choices=output_format_names,
                        help=f'Output format. Defaults to {output_format_names[0]!r}.')
    parser.add_argument('--size', '-s', dest='output_size', default=DEFAULT_output_size,
                        help='Output size in pixels using format "<width>,<height>". '
                             f'Defaults to {DEFAULT_output_size!r}.')
    parser.add_argument('--region', '-r', dest='output_region',
                        help='Output region using format "<lon-min>,<lat-min>,<lon-max>,<lat-max>"')
    parser.add_argument('--meta-file', '-m', dest='output_meta_file',
                        help='File containing cube-level CF-compliant metadata in YAML format.')
    parser.add_argument('--variables', '-v', dest='output_variables',
                        help='Variables to be included in output. Comma-separated list of names.')
    parser.add_argument('--append', '-a', default=False, action='store_true',
                        help='Append successive outputs.')
    parser.add_argument('--dry-run', default=False, action='store_true',
                        help='Just read and process inputs, but don\'t produce any outputs.')
    parser.add_argument('--type', '-t', dest='input_type',
                        default=input_type_names[0], choices=input_type_names,
                        help=f'Input type. Defaults to {input_type_names[0]!r}.')
    parser.add_argument('input_files', metavar='INPUT_FILES', nargs='+',
                        help="One or more input files or a pattern that may contain wildcards '?', '*', and '**'.")

    try:
        arg_obj = parser.parse_args(args or sys.argv[1:])
    except SystemExit as e:
        return int(str(e))

    input_files = arg_obj.input_files
    input_type = arg_obj.input_type
    output_dir = arg_obj.output_dir or DEFAULT_OUTPUT_DIR
    output_name = arg_obj.output_name or DEFAULT_OUTPUT_PATTERN
    output_format = arg_obj.output_format or DEFAULT_OUTPUT_FORMAT
    output_size = arg_obj.output_size
    output_region = arg_obj.output_region
    output_variables = arg_obj.output_variables
    output_meta_file = arg_obj.output_meta_file
    append = arg_obj.append
    dry_run = arg_obj.dry_run

    if output_size:
        try:
            output_size = list(map(lambda c: int(c), output_size.split(',')))
        except ValueError:
            output_size = None
        if output_size is None or len(output_size) != 2:
            print(f'error: invalid size {arg_obj.output_size!r}')
            return 2

    if output_region:
        try:
            output_region = list(map(lambda c: float(c), output_region.split(',')))
        except ValueError:
            output_region = None
        if output_region is None or len(output_region) != 4:
            print(f'error: invalid region {arg_obj.output_region!r}')
            return 2

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
                dst_metadata = load_yaml(stream)
            print(f'loaded metadata from file {arg_obj.output_meta_file!r}')
        except OSError as e:
            print(f'error: failed loading metadata file {arg_obj.output_meta_file!r}: {e}')
            return 2
    else:
        dst_metadata = None

    from xcube.genl2c.process import process_inputs
    process_inputs(input_files,
                   input_type,
                   output_size,
                   output_region,
                   output_variables,
                   dst_metadata,
                   output_dir,
                   output_name,
                   output_format,
                   append,
                   dry_run=dry_run,
                   monitor=print)
    return 0


class GenL2CHelpFormatter(argparse.HelpFormatter):

    def format_help(self):
        # noinspection PyUnresolvedReferences
        help_text = super().format_help()

        ds_io_registry = get_default_dataset_io_registry()

        input_ds_ios = ds_io_registry.query(lambda ds_io: isinstance(ds_io, InputProcessor))
        output_ds_ios = ds_io_registry.query(lambda ds_io: 'w' in ds_io.modes)

        help_text += '\noutput formats to be used with option --format:\n'
        help_text += self._format_dataset_ios(output_ds_ios)
        help_text += '\ninput types to be used with option --type:\n'
        help_text += self._format_dataset_ios(input_ds_ios)

        return help_text

    @classmethod
    def _format_dataset_ios(cls, dataset_ios):
        help_text = ''
        for ds_io in dataset_ios:
            fill1 = ' ' * (24 - len(ds_io.name))
            fill2 = ' ' * (10 - len(ds_io.ext))
            help_text += f'  {ds_io.name}{fill1}(*.{ds_io.ext}){fill2}{ds_io.description}\n'
        return help_text


if __name__ == '__main__':
    sys.exit(status=main())
