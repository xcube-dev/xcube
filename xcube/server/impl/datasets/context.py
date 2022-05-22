# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, List, Dict, Optional

from xcube.server.api import ApiContext
from xcube.server.context import Context
from xcube.constants import LOG


class DatasetsContext(ApiContext):

    def __init__(self, root: Context):
        super().__init__(root)
        self._datasets = dict()

    def update(self, prev_ctx: Optional["DatasetsContext"]):
        LOG.info('Updating datasets...')
        data_store_configs: List[Dict[str, Any]] = \
            self.config.get('data_stores', [])
        for data_store_config in data_store_configs:
            store_id = data_store_config['store_id']
            store_params = data_store_config['store_params']
            dataset_configs: List[Dict[str, Any]] \
                = data_store_config.get('datasets', [])
            for dataset_config in dataset_configs:
                data_id = data_store_config['data_id']
                open_params = data_store_config['open_params']
                print(f'opening {data_id!r} of store {store_id!r}')
