# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import datetime

from xcube.server.api import ApiContext
from xcube.server.api import Context
from xcube.server.server import ServerContext


class MetaContext(ApiContext):
    _start_time = None

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        # API contexts are instantiated every time the config changes
        if self._start_time is None:
            self._start_time = self.now()
        self._update_time = self.now()

    @property
    def server_ctx(self) -> ServerContext:
        server_ctx = super().server_ctx
        assert isinstance(server_ctx, ServerContext)
        return server_ctx

    @property
    def start_time(self) -> str:
        return self._start_time

    @property
    def update_time(self) -> str:
        return self._update_time

    @property
    def current_time(self) -> str:
        return self.now()

    @staticmethod
    def now():
        return datetime.datetime.now().isoformat()
