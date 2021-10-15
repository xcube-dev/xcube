# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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
from typing import Tuple

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.normalize import encode_cube
from xcube.core.store import DataStorePool
from xcube.core.store import get_data_store_instance
from xcube.core.store import new_data_writer
from xcube.util.progress import observe_dask_progress
from ..config import OutputConfig


class CubeWriter:
    def __init__(self,
                 output_config: OutputConfig,
                 store_pool: DataStorePool = None):
        self._output_config = output_config
        self._store_pool = store_pool

    def write_cube(self,
                   cube: xr.Dataset,
                   gm: GridMapping) -> Tuple[str, xr.Dataset]:
        output_config = self._output_config
        dataset = encode_cube(cube, grid_mapping=gm)
        with observe_dask_progress('writing cube', 100):
            write_params = output_config.write_params or {}
            store_params = output_config.store_params or {}
            if output_config.store_id:
                store_instance = get_data_store_instance(
                    output_config.store_id,
                    store_params=store_params,
                    store_pool=self._store_pool
                )
                writer = store_instance.store
                write_params.update(
                    writer_id=output_config.writer_id,
                    **write_params
                )
            else:
                writer = new_data_writer(output_config.writer_id)
                write_params.update(**store_params, **write_params)
            if not dataset.attrs.get('title'):
                # Set fallback title, so we can distinguish
                # datasets from stores in xcube-viewer
                dataset = dataset.assign_attrs(title=output_config.data_id)
            data_id = writer.write_data(
                dataset,
                data_id=output_config.data_id,
                replace=output_config.replace or False,
                **write_params
            )
        return data_id, dataset
