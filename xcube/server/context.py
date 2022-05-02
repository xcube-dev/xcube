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

import abc

import tornado.httpserver

from .config import ServerConfig


class ServerContext(abc.ABC):
    """An abstract context."""

    @property
    @abc.abstractmethod
    def server_config(self) -> ServerConfig:
        """Get the server's configuration."""


class ServerContextImpl(ServerContext):
    """The server context."""

    def __init__(self, server_config: ServerConfig):
        self._server_config = server_config

    @property
    def server_config(self) -> ServerConfig:
        return self._server_config


class RequestContext(ServerContext):
    """A request context."""

    def __init__(self,
                 server_context: ServerContext,
                 request: tornado.httpserver.HTTPRequest):
        self._server_context = server_context
        self._request = request

    @property
    def request(self) -> tornado.httpserver.HTTPRequest:
        return self._request

    @property
    def server_config(self) -> ServerConfig:
        return self._server_context.server_config
