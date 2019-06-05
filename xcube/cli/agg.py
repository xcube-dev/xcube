# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

from xcube.api.resample import resample_in_time, RESAMPLING_METHODS
# TODO (forman): use DatasetIO registry
from xcube.util.config import load_configs

OUTPUT_FORMAT_NAMES = ['zarr', 'nc']

DEFAULT_OUTPUT_DIR = '.'
DEFAULT_OUTPUT_PATTERN = 'L3_{INPUT_FILE}'
DEFAULT_OUTPUT_FORMAT = 'zarr'
DEFAULT_OUTPUT_RESAMPLING_METHOD = 'nearest'
DEFAULT_OUTPUT_FREQUENCY = '1D'


@click.command(name='apply')
@click.argument('input')
@click.option('--config', '-c',
              help='Configuration file in YAML format.')
@click.option('--config', '-c', multiple=True,
              help='Data cube configuration file in YAML format. More than one config input file is allowed.'
                   'When passing several config files, they are merged considering the order passed via command line.')
@click.option('--format', '-f',
              default='zarr',
              type=click.Choice(['zarr', 'nc']),
              help="Output format.")
@click.option('--vars',
              help="Comma-separated list of names of variables to be included.")
@click.option('--resampling', choices=RESAMPLING_METHODS,
              help='Temporal resampling method. Use format "<count><offset>"'
                   'where <offset> is one of {H, D, W, M, Q, Y}'
                   f'Defaults to {DEFAULT_OUTPUT_RESAMPLING_METHOD!r}.')
@click.option('--frequency', dest='output_frequency',
              help='Temporal aggregation frequency.'
                   f'Defaults to {DEFAULT_OUTPUT_FREQUENCY!r}.')
@click.option('--dry-run', default=False, is_flag=True,
              help='Just read and process inputs, but don\'t produce any outputs.')
def agg(
        input,
        config,
        output,
        format,
        vars,
        resampling,
        frequency,
        dry_run,
):
    """
    Perform a temporal aggregation.
    """

    input_path = input
    config_files = config
    output_path = output
    output_format = format
    output_variables = vars
    output_resampling = resampling
    output_frequency = frequency

    config = load_configs(*config_files) if config_files else {}

    if input_path:
        config['input_path'] = input_path
    if output_path:
        config['output_path'] = output_path
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
            print(f'error: invalid variables {arg_obj.output_variables!r}')
            return 2
        # TODO: instead filter list from config
        config['output_variables'] = output_variables

    # noinspection PyBroadException
    _resample_in_time(*config,
                      dry_run=dry_run,
                      monitor=print)

    return 0


def _resample_in_time(input_path: str,
                      output_variables=None,
                      output_metadata=None,
                      output_path=DEFAULT_OUTPUT_DIR,
                      output_format=DEFAULT_OUTPUT_FORMAT,
                      output_resampling=DEFAULT_OUTPUT_RESAMPLING_METHOD,
                      output_frequency=DEFAULT_OUTPUT_FREQUENCY,
                      dry_run=False,
                      monitor=None):
    monitor(f'Reading L2C cube from {input_path!r}...')
    ds = _read_dataset(input_path)

    monitor('Resampling...')
    resampled_ds = resample_in_time(ds, output_frequency, output_resampling, output_variables, output_metadata)

    monitor(f'Writing L3 cube to {output_path!r}...')
    if not dry_run:
        _write_dataset(resampled_ds, output_path, output_format)

    monitor(f'Done.')
