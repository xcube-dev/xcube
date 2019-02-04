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

import click
import yaml

from xcube.genl3.defaults import DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_NAME, DEFAULT_OUTPUT_FORMAT, \
    DEFAULT_OUTPUT_RESAMPLING_METHOD, DEFAULT_OUTPUT_FREQUENCY
from xcube.genl3.process import generate_l3_cube, OUTPUT_FORMAT_NAMES, RESAMPLING_METHODS, FREQUENCY_CHOICES
from xcube.version import version

__import__('xcube.plugin')


@click.command(context_settings={"ignore_unknown_options": True})
@click.version_option(version)
@click.argument('input_files', metavar='INPUT_FILES', nargs=-1)
@click.option('--dir', '-d', metavar='OUTPUT_DIR', default=DEFAULT_OUTPUT_DIR,
              help=f'Output directory. Defaults to {DEFAULT_OUTPUT_DIR!r}')
@click.option('--name', '-n', metavar='OUTPUT_NAME', default=DEFAULT_OUTPUT_NAME,
              help=f'Output filename pattern. Defaults to {DEFAULT_OUTPUT_NAME!r}.')
@click.option('--format', '-f', metavar='OUTPUT_FORMAT', default=DEFAULT_OUTPUT_FORMAT,
              type=click.Choice(OUTPUT_FORMAT_NAMES),
              help=f'Output format. Defaults to {DEFAULT_OUTPUT_FORMAT!r}. '
              f'The choices as output format are: {OUTPUT_FORMAT_NAMES}')
@click.option('--variables', '-v', metavar='OUTPUT_VARIABLES',
              help='Variables to be included in output. '
                   'Comma-separated list of names which may contain wildcard characters "*" and "?".')
@click.option('--resampling', metavar='OUTPUT_RESAMPLING', type=click.Choice(RESAMPLING_METHODS),
              default=DEFAULT_OUTPUT_RESAMPLING_METHOD,
              help='Temporal resampling method. Use format "<count><offset>"'
                   'where <offset> is one of {H, D, W, M, Q, Y}. '
              f'Defaults to {DEFAULT_OUTPUT_RESAMPLING_METHOD!r}. '
              f'The choices for the resampling method are: {RESAMPLING_METHODS}')
@click.option('--frequency', metavar='OUTPUT_FREQUENCY', type=click.Choice(FREQUENCY_CHOICES),
              default=DEFAULT_OUTPUT_FREQUENCY,
              help='Temporal aggregation frequency.'
              f'Defaults to {DEFAULT_OUTPUT_FREQUENCY!r}.'
              f'The choices for the frequency are: {FREQUENCY_CHOICES}')
@click.option('--meta_file', '-m', metavar='OUTPUT_META_FILE',
              help='File containing cube-level, CF-compliant metadata in YAML format.')
@click.option('--dry_run', metavar='DRY_RUN', default=False, is_flag=True,
              help='Just read and process inputs, but don\'t produce any outputs.')
@click.option('--traceback', metavar='TRACEBACK_MODE', default=False, is_flag=True,
              help='On error, print Python traceback.')
def cli(input_files: str,
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
    output_name = name or DEFAULT_OUTPUT_NAME
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
                import traceback
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
            import traceback
            traceback.print_exc()
        return 2

    return 0
