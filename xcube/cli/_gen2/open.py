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

from typing import Sequence, Dict

from xcube.cli._gen2.genconfig import CubeConfig
from xcube.cli._gen2.genconfig import InputConfig
from xcube.cli._gen2.storeconfig import get_data_store_instance
from xcube.core.store import DataStore
from xcube.core.store import new_data_opener
from xcube.util.progress import observe_progress


def open_cubes(input_configs: Sequence[InputConfig],
               cube_config: CubeConfig,
               store_instances: Dict[str, DataStore] = None):
    cubes = []
    all_cube_params = cube_config.to_dict()
    with observe_progress('Opening input(s)', len(input_configs)) as progress:
        for input_config in input_configs:
            open_params = {}
            if input_config.store_id:
                opener = get_data_store_instance(input_config.store_id, input_config.store_params, store_instances)
                open_params.update(opener_id=input_config.opener_id, **input_config.open_params)
            else:
                opener = new_data_opener(input_config.opener_id)
                open_params.update(**input_config.store_params, **input_config.open_params)
            open_params_schema = opener.get_open_data_params_schema(input_config.data_id)
            cube_params = {k: v for k, v in all_cube_params.items() if k in open_params_schema.properties}
            cube = opener.open_data(input_config.data_id, **open_params, **cube_params)
            cubes.append(cube)
            progress.worked(1)

    return cubes
