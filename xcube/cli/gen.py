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

from xcube.api.gen.config import get_config_dict
from xcube.api.gen.defaults import DEFAULT_OUTPUT_PATH, DEFAULT_OUTPUT_RESAMPLING
from xcube.api.gen.gen import gen_cube
from xcube.api.gen.iproc import InputProcessor
from xcube.util.dsio import query_dataset_io
from xcube.util.objreg import get_obj_registry
from xcube.util.reproject import NAME_TO_GDAL_RESAMPLE_ALG

input_processor_names = [input_processor.name
                         for input_processor in get_obj_registry().get_all(type=InputProcessor)]
output_writer_names = [ds_io.name for ds_io in query_dataset_io(lambda ds_io: 'w' in ds_io.modes)]
resampling_algs = NAME_TO_GDAL_RESAMPLE_ALG.keys()


# noinspection PyShadowingBuiltins
@click.command(name='gen', context_settings={"ignore_unknown_options": True})
@click.argument('inputs', nargs=-1)
@click.option('--proc', '-p', type=click.Choice(input_processor_names),
              help=f'Input processor type name. '
                   f'The choices as input processor and additional information about input processors '
                   ' can be accessed by calling xcube gen --info . Defaults to "default" - the default input processor '
                   'that can deal with most common datasets conforming with the CF conventions.')
@click.option('--config', '-c', multiple=True,
              help='Data cube configuration file in YAML format. More than one config input file is allowed.'
                   'When passing several config files, they are merged considering the order passed via command line.')
@click.option('--output', '-o', default=DEFAULT_OUTPUT_PATH,
              help=f'Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}')
@click.option('--format', '-f', type=click.Choice(output_writer_names),
              help=f'Output format. '
                   f'The choices for the output format are: {output_writer_names}.'
                   ' Additional information about output formats can be accessed by calling '
                   'xcube gen --info. If omitted, the format will be guessed from the given output path.')
@click.option('--size', '-s',
              help='Output size in pixels using format "<width>,<height>".')
@click.option('--region', '-r',
              help='Output region using format "<lon-min>,<lat-min>,<lon-max>,<lat-max>"')
@click.option('--variables', '--vars', '-v',
              help='Variables to be included in output. '
                   'Comma-separated list of names which may contain wildcard characters "*" and "?".')
@click.option('--resampling', type=click.Choice(resampling_algs),
              default=DEFAULT_OUTPUT_RESAMPLING,
              help='Fallback spatial resampling algorithm to be used for all variables. '
                   f'Defaults to {DEFAULT_OUTPUT_RESAMPLING!r}. '
                   f'The choices for the resampling algorithm are: {resampling_algs}')
@click.option('--append', '-a', is_flag=True,
              help='Deprecated. The command will now always create, insert, replace, or append input slices.')
@click.option('--prof', is_flag=True,
              help='Collect profiling information and dump results after processing.')
@click.option('--sort', is_flag=True,
              help='The input file list will be sorted before creating the data cube. '
                   'If --sort parameter is not passed, order of input list will be kept.')
@click.option('--info', '-i', is_flag=True,
              help='Displays additional information about format options or about input processors.')
@click.option('--dry_run', is_flag=True,
              help='Just read and process inputs, but don\'t produce any outputs.')
def gen(inputs: str,
        proc: str,
        config: tuple,
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
    Generate data cube.
    Data cubes may be created in one go or successively in append mode, input by input.
    The input paths may be one or more input files or a pattern that may contain wildcards '?', '*', and '**'.
    The input paths can also be passed as lines of a text file. To do so, provide exactly one input file with
    ".txt" extension which contains the actual input paths to be used.
    """
    input_paths = inputs
    input_processor = proc
    config_file = config
    output_path = output
    output_writer = format
    output_size = size
    output_region = region
    output_variables = variables
    output_resampling = resampling
    profile_mode = prof
    dry_run = dry_run
    info_mode = info
    sort_mode = sort

    # Force loading of plugins
    __import__('xcube.util.plugin')

    if info_mode:
        print(_format_info())
        return 0

    config = get_config_dict(locals())

    gen_cube(dry_run=dry_run,
             monitor=print,
             **config)

    return 0


def _format_info():
    # noinspection PyUnresolvedReferences
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
