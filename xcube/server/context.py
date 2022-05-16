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
from typing import Any, Mapping

import tornado.httpserver


class ServerContext(abc.ABC):
    """An abstract context."""

    @property
    @abc.abstractmethod
    def server_config(self) -> Mapping[str, Any]:
        """Get the server's configuration."""

    def get_api_config(self, api_name: str) -> Any:
        """Get the API configuration for *api_name*."""
        return self.server_config.get(api_name)

    @abc.abstractmethod
    def get_api_context(self, api_name: str) -> Any:
        """Get the API context for *api_name*."""


class ServerContextImpl(ServerContext):
    """The server context."""

    def __init__(self, server_config: Mapping[str, Any]):
        self._server_config = dict(server_config)
        self._api_contexts = dict()

    @property
    def server_config(self) -> Mapping[str, Any]:
        return self._server_config

    def set_api_context(self, api_name: str, api_context: Any):
        """Set the API context for *api_name* to *api_context*."""
        self._api_contexts[api_name] = api_context

    def get_api_context(self, api_name: str) -> Any:
        """Get the API context for *api_name*."""
        return self._api_contexts.get(api_name)


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
    def server_config(self) -> Mapping[str, Any]:
        return self._server_context.server_config
