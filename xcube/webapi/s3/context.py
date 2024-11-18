# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Context
from .dsmapping import DatasetsMapping
from .objectstorage import ObjectStorage
from ..datasets.context import DatasetsContext
from xcube.webapi.common.context import ResourcesContext


class S3Context(ResourcesContext):
    """Context for S3 API."""

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self._datasets_ctx = server_ctx.get_api_ctx("datasets")
        self._buckets = {
            "datasets": ObjectStorage(DatasetsMapping(self._datasets_ctx, False)),
            "pyramids": ObjectStorage(
                DatasetsMapping(self._datasets_ctx, True),
            ),
        }

    @property
    def datasets_ctx(self) -> DatasetsContext:
        return self._datasets_ctx

    def get_bucket(self, bucket_name: str) -> ObjectStorage:
        return self._buckets[bucket_name]
