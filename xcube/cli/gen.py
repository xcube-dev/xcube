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
from typing import Sequence

import click

from xcube.api.gen.defaults import DEFAULT_OUTPUT_PATH, DEFAULT_OUTPUT_RESAMPLING
from xcube.util.constants import RESAMPLING_METHOD_NAMES

resampling_methods = sorted(RESAMPLING_METHOD_NAMES)


# noinspection PyShadowingBuiltins
@click.command(name='gen', context_settings={"ignore_unknown_options": True})
@click.argument('input', nargs=-1)
@click.option('--proc', '-P', metavar='INPUT-PROCESSOR', default='default',
              help=f'Input processor name. '
                   f'The available input processor names and additional information about input processors '
                   'can be accessed by calling xcube gen --info . Defaults to "default", an input processor '
                   'that can deal with simple datasets whose variables have dimensions ("lat", "lon") and '
                   'conform with the CF conventions.')
@click.option('--config', '-c', metavar='CONFIG', multiple=True,
              help='xcube dataset configuration file in YAML format. More than one config input file is allowed.'
                   'When passing several config files, they are merged considering the order passed via command line.')
@click.option('--output', '-o', metavar='OUTPUT', default=DEFAULT_OUTPUT_PATH,
              help=f'Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}')
@click.option('--format', '-f', metavar='FORMAT',
              help=f'Output format. '
                   'Information about output formats can be accessed by calling '
                   'xcube gen --info. If omitted, the format will be guessed from the given output path.')
@click.option('--size', '-S', metavar='SIZE',
              help='Output size in pixels using format "<width>,<height>".')
@click.option('--region', '-R', metavar='REGION',
              help='Output region using format "<lon-min>,<lat-min>,<lon-max>,<lat-max>"')
@click.option('--variables', '--vars', metavar='VARIABLES',
              help='Variables to be included in output. '
                   'Comma-separated list of names which may contain wildcard characters "*" and "?".')
@click.option('--resampling', type=click.Choice(resampling_methods),
              default=DEFAULT_OUTPUT_RESAMPLING,
              help='Fallback spatial resampling algorithm to be used for all variables. '
                   f'Defaults to {DEFAULT_OUTPUT_RESAMPLING!r}. '
                   f'The choices for the resampling algorithm are: {resampling_methods}')
@click.option('--append', '-a', is_flag=True,
              help='Deprecated. The command will now always create, insert, replace, or append input slices.')
@click.option('--prof', is_flag=True,
              help='Collect profiling information and dump results after processing.')
@click.option('--sort', is_flag=True,
              help='The input file list will be sorted before creating the xcube dataset. '
                   'If --sort parameter is not passed, order of input list will be kept.')
@click.option('--info', '-I', is_flag=True,
              help='Displays additional information about format options or about input processors.')
@click.option('--dry_run', is_flag=True,
              help='Just read and process inputs, but don\'t produce any outputs.')
def gen(input: Sequence[str],
        proc: str,
        config: Sequence[str],
        output: str,
        format: str,
        size: str,
        region: str,
        variables: str,
        resampling: str,
        append: bool,
        prof: bool,
        dry_run: bool,
        info: bool,
        sort: bool):
    """
    Generate xcube dataset.
    Data cubes may be created in one go or successively in append mode, input by input.
    The input paths may be one or more input files or a pattern that may contain wildcards '?', '*', and '**'.
    The input paths can also be passed as lines of a text file. To do so, provide exactly one input file with
    ".txt" extension which contains the actual input paths to be used.
    """
    dry_run = dry_run
    info_mode = info

    from xcube.api.gen.config import get_config_dict
    from xcube.api.gen.gen import gen_cube
    # Force loading of plugins
    __import__('xcube.util.plugin')

    if info_mode:
        print(_format_info())
        return 0

    config = get_config_dict(
        input_paths=input,
        input_processor_name=proc,
        config_files=config,
        output_path=output,
        output_writer_name=format,
        output_size=size,
        output_region=region,
        output_variables=variables,
        output_resampling=resampling,
        profile_mode=prof,
        append_mode=append,
        sort_mode=sort,
    )

    gen_cube(dry_run=dry_run,
             monitor=print,
             **config)

    return 0


def _format_info():
    from xcube.api.gen.iproc import InputProcessor
    from xcube.util.dsio import query_dataset_io
    from xcube.util.objreg import get_obj_registry

    input_processors = get_obj_registry().get_all(type=InputProcessor)
    output_writers = query_dataset_io(lambda ds_io: 'w' in ds_io.modes)

    help_text = '\ninput processors to be used with option --proc:\n'
    help_text += _format_input_processors(input_processors)
    help_text += '\nFor more input processors use existing "xcube-gen-..." plugins ' \
                 "from the xcube's GitHub organisation or write your own plugin.\n"
    help_text += '\n'
    help_text += '\noutput formats to be used with option --format:\n'
    help_text += _format_dataset_ios(output_writers)
    help_text += '\n'

    return help_text


def _format_input_processors(input_processors):
    help_text = ''
    for input_processor in input_processors:
        fill = ' ' * (34 - len(input_processor.name))
        help_text += f'  {input_processor.name}{fill}{input_processor.description}\n'
    return help_text


def _format_dataset_ios(dataset_ios):
    help_text = ''
    for ds_io in dataset_ios:
        fill1 = ' ' * (24 - len(ds_io.name))
        fill2 = ' ' * (10 - len(ds_io.ext))
        help_text += f'  {ds_io.name}{fill1}(*.{ds_io.ext}){fill2}{ds_io.description}\n'
    return help_text
