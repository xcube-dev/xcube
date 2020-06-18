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

from typing import Type

import click

from xcube.cli._gen2.callback import CallbackApiProgressObserver
from xcube.cli._gen2.open import open_cubes
from xcube.cli._gen2.request import Request, OutputConfig
from xcube.cli._gen2.resample import resample_and_merge_cubes
from xcube.cli._gen2.write import write_cube
from xcube.core.store import find_data_writer_extensions
from xcube.core.store import get_data_accessor_predicate
from xcube.util.progress import observe_progress


def main(request_path: str,
         output_path: str = None,
         format_name: str = None,
         callback_api_url: str = None,
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
    :param output_path: output ZARR directory in local file system.
        Overwrites output configuration in request if given.
    :param callback_api_url:  Optional API URL used to report status information. The URL
        must accept the PUT method and support the JSON content type.
    :param verbose:
    :param exception_type: exception type used to raise on errors
    """

    if callback_api_url:
        CallbackApiProgressObserver(callback_api_url).activate()

    request = Request.from_file(request_path, exception_type=exception_type)

    if output_path:
        output_config = _new_output_config_for_dir(output_path, format_name, exception_type)
    else:
        output_config = request.output_config

    with observe_progress('Generating cube', 100) as cm:
        cm.will_work(10)
        cubes = open_cubes(request.input_configs,
                           cube_config=request.cube_config)
        cm.will_work(10)
        cube = resample_and_merge_cubes(cubes,
                                        cube_config=request.cube_config)
        cm.will_work(80)
        write_cube(cube,
                   output_config=output_config)


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
