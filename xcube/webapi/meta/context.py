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

