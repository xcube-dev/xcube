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
import click

from xcube.dsio import query_dataset_io
from xcube.genl2c.config import get_config_dict
from xcube.genl2c.defaults import DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_NAME, \
    DEFAULT_OUTPUT_RESAMPLING, DEFAULT_OUTPUT_WRITER
from xcube.genl2c.inputprocessor import InputProcessor
from xcube.genl2c.process import generate_l2c_cube
from xcube.objreg import get_obj_registry
from xcube.reproject import NAME_TO_GDAL_RESAMPLE_ALG
from xcube.version import version

input_processor_names = [input_processor.name
                         for input_processor in get_obj_registry().get_all(type=InputProcessor)]
output_writer_names = [ds_io.name for ds_io in query_dataset_io(lambda ds_io: 'w' in ds_io.modes)]
resampling_algs = NAME_TO_GDAL_RESAMPLE_ALG.keys()

@click.command(context_settings={"ignore_unknown_options":True})
@click.argument('input_files', metavar='INPUT_FILES', nargs=-1)
@click.option('--proc', '-p', metavar='INPUT_PROCESSOR', type=click.Choice(input_processor_names),
              help=f'Input processor type name. '
              f'The choices as input processor are: {input_processor_names}')
@click.option('--config', '-c', metavar='CONFIG_FILE',
              help='Data cube configuration file in YAML format.')
@click.option('--dir', '-d', metavar='OUTPUT_DIR', default=str(DEFAULT_OUTPUT_DIR),
              help=f'Output directory. Defaults to {DEFAULT_OUTPUT_DIR!r}')
@click.option('--name', '-n', metavar='OUTPUT_NAME', default=str(DEFAULT_OUTPUT_NAME),
              help=f'Output filename pattern. Defaults to {DEFAULT_OUTPUT_NAME!r}.')
@click.option('--format', '-f', metavar='OUTPUT_FORMAT', type=click.Choice(output_writer_names),
              default={DEFAULT_OUTPUT_WRITER}, help=f'Output writer type name. Defaults to {DEFAULT_OUTPUT_WRITER!r}. '
              f'The choices for the output format are: {output_writer_names}')
@click.option('--size', '-s', metavar='OUTPUT_SIZE',
              help='Output size in pixels using format "<width>,<height>".')
@click.option('--region', '-r', metavar='OUTPUT_REGION',
              help='Output region using format "<lon-min>,<lat-min>,<lon-max>,<lat-max>"')
@click.option('--variables', '-v', metavar='OUTPUT_VARIABLES',
              help='Variables to be included in output. '
                   'Comma-separated list of names which may contain wildcard characters "*" and "?".')
@click.option('--resampling', metavar='OUTPUT_RESAMPLING', type=click.Choice(resampling_algs),
              help='Fallback spatial resampling algorithm to be used for all variables. '
              f'Defaults to {DEFAULT_OUTPUT_RESAMPLING!r}. '
              f'The choices for the resampling algorithm are: {resampling_algs}')
@click.option('--traceback', metavar='TRACEBACK_MODE', default=False, is_flag=True,
              help='On error, print Python traceback.')
@click.option('--append', '-a', metavar='APPEND_MODE', default=False, is_flag=True,
              help='Append successive outputs.')
@click.option('--dry_run', metavar='DRY_RUN', default=False, is_flag=True,
              help='Just read and process inputs, but don\'t produce any outputs.')
def create_xcube(input_files: str,
                 proc: str,
                 config: str,
                 dir: str,
                 name: str,
                 format: str,
                 size: str ,
                 region: str,
                 variables: str,
                 resampling: str,
                 traceback: bool,
                 append: bool,
                 dry_run: bool):
    """
    Generate or extend a Level-2C data cube from Level-2 input files.
    Level-2C data cubes may be created in one go or in successively
    in append mode, input by input.

    The input may be one or more input files or a pattern that may contain wildcards '?', '*', and '**'.
    """


    traceback_mode = traceback
    append_mode = append
    dry_run = dry_run
    try:
        config = get_config_dict(locals(), open)
    except ValueError as e:
        return _handle_error(e, traceback_mode)

    traceback_mode = traceback
    # noinspection PyBroadException
    try:
        generate_l2c_cube(append_mode=append_mode,
                          dry_run=dry_run,
                          monitor=print,
                          **config)

    except Exception as e:
        return _handle_error(e, traceback_mode)

    return 0


def _handle_error(e, traceback_mode):
    print(f'error: {e}', file=sys.stderr)
    if traceback_mode:
        traceback.print_exc(file=sys.stderr)
    return 2

@click.group()
@click.version_option(version)
def cli():
    """
    Generate or extend a Level-2C data cube from Level-2 input files.
    Level-2C data cubes may be created in one go or in successively in append mode, input by input.
   """


cli.add_command(create_xcube)

