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

import os.path

import click
from typing import Type

from xcube.cli._gen2.open import open_cubes
from xcube.cli._gen2.request import Request, OutputConfig
from xcube.cli._gen2.resample import resample_and_merge_cubes
from xcube.cli._gen2.transform import transform_cube
from xcube.cli._gen2.write import write_cube
from xcube.core.dsio import guess_dataset_format

DEFAULT_GEN_OUTPUT_PATH = 'out.zarr'


def main(request_path: str,
         output_path: str = None,
         callback_url=None,
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
        If no output configuration is given in request, *output_path* defaults to "out.zarr".
    :param callback_url:  Optional URL used to status information. The URL
        must accept the POST method and support the JSON content type.
    :param verbose:
    :param exception_type: exception type used to raise on errors
    """

    def progress_monitor():
        # TODO: make use of callback_url and verbose
        pass

    request = Request.from_file(request_path, exception_type=exception_type)

    if output_path:
        base_dir = os.path.dirname(output_path)
        cube_id, _ = os.path.splitext(os.path.basename(output_path))
        output_config = OutputConfig(cube_store_id='dir',
                                     cube_store_params=dict(base_dir=base_dir, read_only=False),
                                     cube_id=cube_id,
                                     write_params=dict(format=guess_dataset_format(output_path)))
    else:
        output_config = request.output_config

    # Step 1
    cubes = open_cubes(request.input_configs,
                       cube_config=request.cube_config,
                       progress_monitor=progress_monitor)
    # Step 2
    cube = resample_and_merge_cubes(cubes,
                                    cube_config=request.cube_config,
                                    progress_monitor=progress_monitor)
    # Step 3
    cube = transform_cube(cube,
                          request.code_config,
                          progress_monitor=progress_monitor)
    # Step 4
    write_cube(cube,
               output_config=output_config,
               progress_monitor=progress_monitor)
