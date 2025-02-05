# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiContext, Context

from ..datasets.context import DatasetsContext


class TimeSeriesContext(ApiContext):
    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self._datasets_ctx = server_ctx.get_api_ctx("datasets", cls=DatasetsContext)

    @property
    def datasets_ctx(self) -> DatasetsContext:
        return self._datasets_ctx
