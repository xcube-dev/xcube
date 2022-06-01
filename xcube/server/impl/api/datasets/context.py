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

import fnmatch
from typing import Any, List, Dict, Optional

from xcube.constants import LOG
from xcube.core.store import new_data_store
from xcube.server.api import ApiContext
from xcube.server.api import ServerContext


class DatasetsContext(ApiContext):

    def __init__(self, root: ServerContext):
        super().__init__(root)
        self._datasets = dict()

    def on_update(self, prev_ctx: Optional["DatasetsContext"]):
        LOG.info('Updating datasets...')
        data_store_configs: List[Dict[str, Any]] = \
            self.config.get('data_stores', [])
        for data_store_config in data_store_configs:
            store_id = data_store_config['store_id']
            store_params = data_store_config.get('store_params', {})
            store = new_data_store(store_id, **store_params)
            dataset_configs: List[Dict[str, Any]] \
                = data_store_config.get('datasets', [])
            available_data_ids = list(store.get_data_ids())
            for dataset_config in dataset_configs:
                data_id = dataset_config['data_id']
                if _is_wildcard(data_id):
                    available_data_ids = list(store.get_data_ids())
                    break

            dataset_resources = []
            for dataset_config in dataset_configs:
                data_id = dataset_config['data_id']
                if _is_wildcard(data_id):
                    for available_data_id in available_data_ids:
                        if fnmatch.fnmatch(available_data_id, data_id):
                            _collect_dataset_resource(
                                store,
                                dataset_config,
                                available_data_id,
                                dataset_resources
                            )
                else:
                    _collect_dataset_resource(
                        store,
                        dataset_config,
                        data_id,
                        dataset_resources
                    )


def _collect_dataset_resource(store,
                              dataset_config,
                              data_id,
                              dataset_resources):
    try:
        data_descriptor = store.describe_data(data_id)
    except Exception as e:
        LOG.error(f'{e}', exc_info=1)
        LOG.warning(f'Skipping data resource {data_id}'
                    f' of store {store.__class__.__name__}'
                    f' due to error above')
        return
    LOG.info(f'Loaded data resource {data_id}'
             f' of store {store.__class__.__name__}')
    dataset_resource = dict(dataset_config)
    dataset_resource['data_id'] = data_id
    dataset_resource['data_descriptor'] = data_descriptor
    dataset_resources.append(dataset_resource)


def _is_wildcard(data_id):
    return '?' in data_id or '*' in data_id
