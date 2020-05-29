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

from xcube.cli._gen2.open import open_cubes
from xcube.cli._gen2.request import Request
from xcube.cli._gen2.resample import RESAMPLE_PARAMS
from xcube.cli._gen2.resample import resample_and_merge_cubes
from xcube.cli._gen2.transform import TRANSFORM_PARAMS
from xcube.cli._gen2.transform import transform_cube
from xcube.cli._gen2.write import write_cube
from xcube.core.store.store import CubeStoreRegistry

DEFAULT_GEN_OUTPUT_PATH = 'out.zarr'


def main(request_path: str,
         output_path: str = None,
         callback_url=None,
         exception_type: Type[BaseException] = click.ClickException,
         cube_store_registry: CubeStoreRegistry = None,
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
    :param cube_store_registry:
    :return:
    """

    request = Request.from_file(request_path, exception_type=exception_type)

    input_configs = [request.input_config]
    resample_params = RESAMPLE_PARAMS.from_json_values(request.cube_config,
                                                       exception_type=exception_type)
    transform_params = TRANSFORM_PARAMS.from_json_values(request.code_config,
                                                         exception_type=exception_type)
    write_params = request.output_config

    # 1.
    cubes = open_cubes(input_configs, resample_params, cube_store_registry, exception_type)
    # 2.
    cube = resample_and_merge_cubes(cubes, resample_params)
    # 3.
    cube = transform_cube(cube, **transform_params)
    # 4.
    write_cube(cube, output_path=output_path, write_params=write_params)
