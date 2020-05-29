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
from typing import Sequence, Type

from xcube.core.store.param import ParamValues
from xcube.core.store.store import CubeStoreRegistry


def open_cubes(input_configs: Sequence[ParamValues],
               cube_config: ParamValues,
               cube_store_registry: CubeStoreRegistry,
               exception_type: Type[BaseException]):
    cube_store_registry = cube_store_registry or CubeStoreRegistry.default()

    cubes = []
    for input_config in input_configs:
        cube_store_id = input_config.cube_store_id
        cube_store = cube_store_registry.get(cube_store_id)
        if cube_store is None:
            raise exception_type(f'Unknown cube store "{cube_store_id}"')
        cube_store_params = cube_store.get_cube_service_params().from_json_values(input_config.cube_store_params,
                                                                                  exception_type=exception_type)
        cube_service = cube_store.new_cube_service(cube_store_params)

        dataset_id = input_config.dataset_id
        open_params = cube_service.get_open_cube_params(dataset_id).from_json_values(input_config.open_params,
                                                                                     exception_type=exception_type)

        cube = cube_service.open_cube(dataset_id, open_params=open_params, cube_params=cube_config)
        cubes.append(cube)

    return cubes
