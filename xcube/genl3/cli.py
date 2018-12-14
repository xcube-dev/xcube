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

import sys
import traceback
from typing import List, Optional
import click
import yaml

from xcube.genl3.defaults import DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_PATTERN, DEFAULT_OUTPUT_FORMAT, \
    DEFAULT_OUTPUT_RESAMPLING_METHOD, DEFAULT_OUTPUT_FREQUENCY
from xcube.genl3.process import generate_l3_cube, OUTPUT_FORMAT_NAMES, RESAMPLING_METHODS, FREQUENCY_CHOICES
from xcube.version import version

__import__('xcube.plugin')


@click.command(context_settings={"ignore_unknown_options":True})
@click.argument('input_files', metavar='INPUT_FILES', nargs=-1)
@click.option('--dir', '-d', metavar='OUTPUT_DIR', default={DEFAULT_OUTPUT_DIR},
              help=f'Output directory. Defaults to {DEFAULT_OUTPUT_DIR!r}')
@click.option('--name', '-n', metavar='OUTPUT_NAME', default={DEFAULT_OUTPUT_PATTERN},
              help=f'Output filename pattern. Defaults to {DEFAULT_OUTPUT_PATTERN!r}.')
@click.option('--format', '-f', metavar='OUTPUT_FORMAT',default={DEFAULT_OUTPUT_FORMAT},
              type=click.Choice(OUTPUT_FORMAT_NAMES), help=f'Output format. Defaults to {DEFAULT_OUTPUT_FORMAT!r}. '
              f'The choices as output format are: {OUTPUT_FORMAT_NAMES}')
@click.option('--variables', '-v', metavar='OUTPUT_VARIABLES',
              help='Variables to be included in output. '
                   'Comma-separated list of names which may contain wildcard characters "*" and "?".')
@click.option('--resampling', metavar='OUTPUT_RESAMPLING', type=click.Choice(RESAMPLING_METHODS),
              help='Temporal resampling method. Use format "<count><offset>"'
                   'where <offset> is one of {H, D, W, M, Q, Y}. '               
              f'Defaults to {DEFAULT_OUTPUT_RESAMPLING_METHOD!r}. '
              f'The choices for the resampling method are: {RESAMPLING_METHODS}')
@click.option('--frequency', metavar='OUTPUT_FREQUENCY', type=click.Choice(FREQUENCY_CHOICES),
              help='Temporal aggregation frequency.'
              f'Defaults to {DEFAULT_OUTPUT_FREQUENCY!r}.'
              f'The choices for the frequency are: {FREQUENCY_CHOICES}')
@click.option('--meta_file', '-m', metavar='OUTPUT_META_FILE',
              help='File containing cube-level, CF-compliant metadata in YAML format.')
@click.option('--dry_run', metavar='DRY_RUN', default=False, is_flag=True,
              help='Just read and process inputs, but don\'t produce any outputs.')
@click.option('--traceback', metavar='TRACEBACK_MODE', default=False, is_flag=True,
              help='On error, print Python traceback.')
def create_xcube_3(input_files: str,
                 dir: str,
                 name: str,
                 format: str,
                 variables: str,
                 resampling: str,
                 frequency: str,
                 meta_file: str,
                 traceback: bool,
                 dry_run: bool):
    """
    Generate L2C data cubes from L2 data products.
    """

    config_file = meta_file
    input_file = input_files
    output_dir = dir or DEFAULT_OUTPUT_DIR
    output_name = name or DEFAULT_OUTPUT_PATTERN
    output_format = format or DEFAULT_OUTPUT_FORMAT
    output_variables = variables
    output_resampling = resampling or DEFAULT_OUTPUT_RESAMPLING_METHOD
    output_frequency = frequency or DEFAULT_OUTPUT_FREQUENCY
    traceback_mode = traceback
    dry_run = dry_run

    if config_file:
        try:
            with open(config_file) as stream:
                config = yaml.load(stream)
            print(f'loaded configuration from {config_file!r}')
        except OSError as e:
            print(f'error: failed loading configuration from {config_file!r}: {e}')
            if traceback_mode:
                traceback.print_exc()
            return 2
    else:
        config = {}

    if input_file:
        config['input_file'] = input_file
    if output_dir:
        config['output_dir'] = output_dir
    if output_name:
        config['output_name'] = output_name
    if output_format:
        config['output_format'] = output_format
    if output_resampling:
        config['output_resampling'] = output_resampling
    if output_frequency:
        config['output_frequency'] = output_frequency
    if output_variables:
        try:
            output_variables = set(map(lambda c: str(c).strip(), output_variables.split(',')))
        except ValueError:
            output_variables = None
        if output_variables is not None \
                and next(iter(True for var_name in output_variables if var_name == ''), False):
            output_variables = None
        if output_variables is None or len(output_variables) == 0:
            print(f'error: invalid variables {variables!r}')
            return 2
        # TODO: instead filter list from config
        config['output_variables'] = output_variables
    # noinspection PyBroadException

    try:
        generate_l3_cube(*config,
                         dry_run=dry_run,
                         monitor=print)
    except BaseException as e:
        print(f'error: {e}')
        if traceback_mode:
            traceback.print_exc()
        return 2

    return 0



@click.group()
@click.version_option(version)
def cli():
    """
    Generate Level-3 data cube from Level-2C data cube.
    The input file or directory which must be a Level-2C cube.
   """

cli.add_command(create_xcube_3)


# def main(args: Optional[List[str]] = None):
#     """
#     Generate L2C data cubes from L2 data products.
#     """
#
#     parser = argparse.ArgumentParser(description='Generate Level-3 data cube from Level-2C data cube.')
#     parser.add_argument('--version', '-V', action='version', version=version)
#     parser.add_argument('--dir', '-d', dest='output_dir', default=DEFAULT_OUTPUT_DIR,
#                         help=f'Output directory. Defaults to {DEFAULT_OUTPUT_DIR!r}')
#     parser.add_argument('--name', '-n', dest='output_name', default=DEFAULT_OUTPUT_PATTERN,
#                         help=f'Output filename pattern. Defaults to {DEFAULT_OUTPUT_PATTERN!r}.')
#     parser.add_argument('--format', '-f', dest='output_format',
#                         default=DEFAULT_OUTPUT_FORMAT, choices=OUTPUT_FORMAT_NAMES,
#                         help=f'Output format. Defaults to {DEFAULT_OUTPUT_FORMAT!r}.')
#     parser.add_argument('--variables', '-v', dest='output_variables',
#                         help='Variables to be included in output. '
#                              'Comma-separated list of names which may contain wildcard characters "*" and "?".')
#     parser.add_argument('--resampling', dest='output_resampling', choices=RESAMPLING_METHODS,
#                         help='Temporal resampling method. Use format "<count><offset>"'
#                              'where <offset> is one of {H, D, W, M, Q, Y}'
#                              f'Defaults to {DEFAULT_OUTPUT_RESAMPLING_METHOD!r}.')
#     parser.add_argument('--frequency', dest='output_frequency',
#                         help='Temporal aggregation frequency.'
#                              f'Defaults to {DEFAULT_OUTPUT_FREQUENCY!r}.')
#     parser.add_argument('--meta-file', '-m', dest='output_meta_file',
#                         help='File containing cube-level, CF-compliant metadata in YAML format.')
#     parser.add_argument('--dry-run', default=False, action='store_true',
#                         help='Just read and process inputs, but don\'t produce any outputs.')
#     parser.add_argument('--traceback', dest='traceback_mode', default=False, action='store_true',
#                         help='On error, print Python traceback.')
#     parser.add_argument('input_file', metavar='INPUT_FILE',
#                         help="The input file or directory which must be a Level-2C cube.")
#
#     try:
#         arg_obj = parser.parse_args(args or sys.argv[1:])
#     except SystemExit as e:
#         return int(str(e))
#
#     config_file = arg_obj.output_meta_file
#     input_file = arg_obj.input_file
#     output_dir = arg_obj.output_dir or DEFAULT_OUTPUT_DIR
#     output_name = arg_obj.output_name or DEFAULT_OUTPUT_PATTERN
#     output_format = arg_obj.output_format or DEFAULT_OUTPUT_FORMAT
#     output_variables = arg_obj.output_variables
#     output_resampling = arg_obj.output_resampling or DEFAULT_OUTPUT_RESAMPLING_METHOD
#     output_frequency = arg_obj.output_frequency or DEFAULT_OUTPUT_FREQUENCY
#     traceback_mode = arg_obj.traceback_mode
#     dry_run = arg_obj.dry_run
#
#     if config_file:
#         try:
#             with open(config_file) as stream:
#                 config = yaml.load(stream)
#             print(f'loaded configuration from {config_file!r}')
#         except OSError as e:
#             print(f'error: failed loading configuration from {config_file!r}: {e}')
#             if traceback_mode:
#                 traceback.print_exc()
#             return 2
#     else:
#         config = {}
#
#     if input_file:
#         config['input_file'] = input_file
#     if output_dir:
#         config['output_dir'] = output_dir
#     if output_name:
#         config['output_name'] = output_name
#     if output_format:
#         config['output_format'] = output_format
#     if output_resampling:
#         config['output_resampling'] = output_resampling
#     if output_frequency:
#         config['output_frequency'] = output_frequency
#     if output_variables:
#         try:
#             output_variables = set(map(lambda c: str(c).strip(), output_variables.split(',')))
#         except ValueError:
#             output_variables = None
#         if output_variables is not None \
#                 and next(iter(True for var_name in output_variables if var_name == ''), False):
#             output_variables = None
#         if output_variables is None or len(output_variables) == 0:
#             print(f'error: invalid variables {arg_obj.output_variables!r}')
#             return 2
#         # TODO: instead filter list from config
#         config['output_variables'] = output_variables
#
#     # noinspection PyBroadException
#     try:
#         generate_l3_cube(*config,
#                          dry_run=dry_run,
#                          monitor=print)
#     except BaseException as e:
#         print(f'error: {e}')
#         if traceback_mode:
#             traceback.print_exc()
#         return 2
#
#     return 0
#
#
# if __name__ == '__main__':
#     sys.exit(status=main())
