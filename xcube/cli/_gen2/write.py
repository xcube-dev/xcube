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
import xarray as xr
from typing import Callable, Optional

from xcube.cli._gen2.request import OutputConfig
from xcube.core.store.helpers import new_cube_store
from xcube.core.store.store import CubeWriter
from xcube.util.extension import ExtensionRegistry


def write_cube(cube: xr.Dataset,
               output_config: OutputConfig,
               progress_monitor: Callable,
               extension_registry: Optional[ExtensionRegistry] = None):
    cube_store = new_cube_store(output_config.cube_store_id,
                                output_config.cube_store_params,
                                cube_store_requirements=(CubeWriter,),
                                extension_registry=extension_registry)
    cube_id = output_config.cube_id
    write_params_schema = cube_store.get_write_cube_params_schema()
    write_params = write_params_schema.from_json(output_config.write_params) \
        if output_config.write_params else {}
    cube_store.write_cube(cube, cube_id=cube_id, write_params=write_params)
