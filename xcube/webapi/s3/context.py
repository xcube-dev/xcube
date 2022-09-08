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

import os

from xcube.server.api import Context
from xcube.webapi.context import FS_TYPE_TO_PROTOCOL
from .config import S3_MODE_DATASET
from .dsmapping import DatasetsMapping
from .objectstorage import ObjectStorage
from ..datasets.context import DatasetsContext
from ..resctx import ResourcesContext


class S3Context(ResourcesContext):
    """Context for S3 API."""

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self._datasets_ctx = server_ctx.get_api_ctx("datasets")
        self._object_storage = ObjectStorage(
            DatasetsMapping(
                self._datasets_ctx,
                server_ctx.config.get("S3", {}).get("Mode", S3_MODE_DATASET)
            )
        )

    @property
    def datasets_ctx(self) -> DatasetsContext:
        return self._datasets_ctx

    @property
    def object_storage(self) -> ObjectStorage:
        return self._object_storage

    # TODO (forman): get rid of this,
    #   we'll use self.object_storage from now on
    def get_s3_bucket_mapping(self):
        s3_bucket_mapping = {}
        for dataset_config in self.datasets_ctx.get_dataset_configs():
            ds_id = dataset_config.get('Identifier')
            protocol = FS_TYPE_TO_PROTOCOL.get(
                dataset_config.get('FileSystem',
                                   'file'))
            if protocol == 'file':
                store_instance_id = dataset_config.get('StoreInstanceId')
                if store_instance_id:
                    data_store_pool = self.datasets_ctx.get_data_store_pool()
                    store_root = data_store_pool.get_store_config(
                        store_instance_id). \
                        store_params.get('root')
                    data_id = dataset_config.get('Path')
                    local_path = os.path.join(store_root, data_id)
                else:
                    local_path = self.get_config_path(dataset_config,
                                                      f'dataset configuration'
                                                      f' {ds_id!r}')
                local_path = os.path.normpath(local_path)
                if os.path.isdir(local_path):
                    s3_bucket_mapping[ds_id] = local_path
        return s3_bucket_mapping
