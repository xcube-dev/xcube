# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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
from typing import Type, Sequence

import click

from xcube.cli._gen2.genconfig import GenConfig, OutputConfig
from xcube.cli._gen2.open import open_cubes
from xcube.cli._gen2.progress import ApiProgressCallbackObserver
from xcube.cli._gen2.resample import resample_and_merge_cubes
from xcube.cli._gen2.write import write_cube
from xcube.core.store import find_data_writer_extensions
from xcube.core.store import get_data_accessor_predicate
from xcube.util.progress import observe_progress, ProgressObserver, ProgressState


def main(request_path: str,
         exception_type: Type[BaseException] = click.ClickException,
         verbose: bool = False):
    """
    Generate a data cube.

    Creates cube views from one or more cube stores, resamples them to a common grid,
    optionally performs some cube transformation,
    and writes the resulting cube to some target cube store.

    REQUEST is the cube generation request. It may be provided as a JSON or YAML file
    (file extensions ".json" or ".yaml"). If the REQUEST file argument is omitted, it is expected that
    the Cube generation request is piped as a JSON string.

    :param request_path: cube generation request. It may be provided as a JSON or YAML file
        (file extensions ".json" or ".yaml"). If the REQUEST file argument is omitted, it is expected that
        the Cube generation request is piped as a JSON string.
    :param verbose:
    :param exception_type: exception type used to raise on errors
    """

    request = GenConfig.from_file(request_path, exception_type=exception_type)

    if request.callback_config:
        ApiProgressCallbackObserver(request.callback_config).activate()
    else:
        ConsoleProgressObserver().activate()

    with observe_progress('Generating cube', 100) as cm:
        cm.will_work(10)
        cubes = open_cubes(request.input_configs,
                           cube_config=request.cube_config)

        cm.will_work(10)
        cube = resample_and_merge_cubes(cubes,
                                        cube_config=request.cube_config)

        cm.will_work(80)
        write_cube(cube,
                   output_config=request.output_config)


def _new_output_config_for_dir(output_path, format_id, exception_type: Type[BaseException]):
    predicate = get_data_accessor_predicate(type_id='dataset', format_id=format_id, data_id=output_path)
    extensions = find_data_writer_extensions(predicate=predicate)
    if not extensions:
        raise exception_type(f'Failed to guess writer from path {output_path}')
    writer_id = extensions[0].name
    output_config = OutputConfig(writer_id=writer_id,
                                 data_id=output_path,
                                 write_params=dict())
    return output_config


class ConsoleProgressObserver(ProgressObserver):

    def on_begin(self, state_stack: Sequence[ProgressState]):
        print(self._format_progress(state_stack, marker='...'))

    def on_update(self, state_stack: Sequence[ProgressState]):
        print(self._format_state_stack(state_stack))

    def on_end(self, state_stack: Sequence[ProgressState]):
        if state_stack[0].exc_info:
            print(self._format_progress(state_stack, marker='error!'))
            if len(state_stack) == 1:
                print(state_stack[0].exc_info_text)
        else:
            print(self._format_progress(state_stack, marker='done.'))

    @classmethod
    def _format_progress(cls, state_stack: Sequence[ProgressState], marker=None) -> str:
        if marker:
            state_stack_part = cls._format_state_stack(state_stack[0:-1])
            state_part = cls._format_state(state_stack[-1], marker=marker)
            return state_part if not state_stack_part else state_stack_part + ': ' + state_part
        else:
            return cls._format_state_stack(state_stack)

    @classmethod
    def _format_state_stack(cls, state_stack: Sequence[ProgressState], marker=None) -> str:
        return ': '.join([cls._format_state(s) for s in state_stack])

    @classmethod
    def _format_state(cls, state: ProgressState, marker=None) -> str:
        if marker is None:
            return '{a} - {b:3.1f}%'.format(a=state.label, b=100 * state.progress)
        else:
            return '{a} - {b}'.format(a=state.label, b=marker)
