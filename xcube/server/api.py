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
from typing import Any, List, Type, Optional

import tornado.httputil
import tornado.web

from .config import ServerConfig, Config
from .context import ServerContext, RequestContext


class ServerConfigExt:
    """A server configuration extension."""

    def __init__(self, server_config: ServerConfig):
        self._server_config = server_config

    @property
    def server_config(self) -> ServerConfig:
        """The server configuration."""
        return self._server_config


class ServerContextExt:
    """An extension for a server context."""

    def __init__(self, server_context: ServerContext, config_ext: ServerConfigExt):
        self._server_context = server_context
        self._config_ext = config_ext

    @property
    def server_context(self) -> ServerContext:
        return self._server_context


class ServerApi(abc.ABC):
    """An abstract server API."""

    def get_handlers(self) -> List[Any]:
        """Get the server API's handlers."""
        return []

    def get_config_class(self) -> Optional[Type[Config]]:
        """
        Get the server API's configuration class
        or None if the API doesn't require configuration.
        """
        return None


class RequestHandler(tornado.web.RequestHandler, abc.ABC):
    def __init__(self,
                 application: tornado.web.Application,
                 request: tornado.httputil.HTTPServerRequest,
                 **kwargs: Any):
        super().__init__(application, request, **kwargs)
        server_context = getattr(application, '__server_context', None)
        if server_context is None:
            raise RuntimeError('request handler must be used with xcube server')
        self._context = RequestContext(server_context, request)

    @property
    def context(self) -> RequestContext:
        return self._context
