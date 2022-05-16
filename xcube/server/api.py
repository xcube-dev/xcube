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
from typing import Any, List, Optional, Tuple, Dict, Union, Type, Sequence, Generic, TypeVar, Mapping

import tornado.httputil
import tornado.ioloop
import tornado.web

from .context import RequestContext
from .context import ServerContext
from ..util.jsonschema import JsonSchema

SERVER_CONTEXT_ATTR_NAME = '__xcube_server_context'

ServerApiRoute = Union[
    Tuple[str, Type["RequestHandler"]],
    Tuple[str, Type["RequestHandler"], Dict[str, Any]]
]

# ConteXt type variable
X = TypeVar("X", bound="ServerApi")


class ServerApi(Generic[X]):
    """
    A server API.

    The most common purpose of this class is to
    add a new API to the server by the means of routes.

    However, an API may be just programmatic and provide
    the context for other APIs.

    May be derived by clients to override the methods

    * `on_start`,
    * `on_stop`,
    * `on_config_change`.

    :param routes: Optional list of routes.
    :param config_schema: Optional JSON schema for the API configuration.
    """

    def __init__(self,
                 name: str, /,
                 dependencies: Optional[Sequence[str]] = None,
                 routes: Optional[Sequence[ServerApiRoute]] = None,
                 config_schema: Optional[JsonSchema] = None):
        self._name = name
        self._dependencies = tuple(dependencies or ())
        self._routes: List[ServerApiRoute] = list(routes or [])
        self._config_schema = config_schema

    @property
    def name(self) -> str:
        """The name of this API."""
        return self._name

    @property
    def dependencies(self) -> Tuple[str]:
        """The names of other APIs on which this API depends on."""
        return self._dependencies

    def route(self, pattern: str, **target_kwargs):
        """
        Decorator that adds a route to this API.

        The decorator target must be a classes
        derived from RequestHandler.

        :param pattern: The route pattern.
        :param target_kwargs: Optional keyword arguments passed to
            RequestHandler constructor.
        :return: A decorator function that receives a
            class derived from RequestHandler
        """

        def decorator_func(target_class: Type["RequestHandler"]):
            if not issubclass(target_class, RequestHandler):
                raise TypeError(f'target_class must be an'
                                f' instance of {RequestHandler},'
                                f' but was {target_class}')
            if target_kwargs:
                handler = pattern, target_class, target_kwargs
            else:
                handler = pattern, target_class
            self._routes.append(handler)

        return decorator_func

    @property
    def routes(self) -> List[ServerApiRoute]:
        """The handlers provided by this API."""
        return self._routes

    @property
    def config_schema(self) -> Optional[JsonSchema]:
        """
        Get the JSON schema for the configuration of this API.
        """
        return self._config_schema

    def on_start(self,
                 server_context: ServerContext,
                 io_loop: tornado.ioloop.IOLoop):
        """
        Called when the server is started.

        :param server_context: The current server context
        :param io_loop: The current i/o loop.
        """

    def on_stop(self,
                server_context: ServerContext,
                io_loop: tornado.ioloop.IOLoop):
        """
        Called when the server is stopped.

        :param server_context: The current server context
        :param io_loop: The current i/o loop.
        """

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def get_context(
            self,
            next_api_config: Any,
            prev_api_context: Optional[X],
            next_server_config: Mapping[str, Any],
            prev_server_context: Optional[ServerContext]
    ) -> X:
        """
        Called when the configuration has changed.

        May return an updated API context for new server configuration
        *next_server_config* and optional previous server
        context *prev_server_context*.

        The default implementation returns the API configuration
        from *next_server_config*.

        :param next_api_config: The new API configuration
        :param prev_api_context: The previous API context
        :param next_server_config: The new server configuration
        :param prev_server_context: Optional previous server context
        :return: an API context object or None
        """
        return getattr(next_server_config, self.name, None)


class RequestHandler(tornado.web.RequestHandler, abc.ABC):
    def __init__(self,
                 application: tornado.web.Application,
                 request: tornado.httputil.HTTPServerRequest,
                 **kwargs: Any):
        super().__init__(application, request, **kwargs)
        server_context = getattr(application, SERVER_CONTEXT_ATTR_NAME, None)
        if server_context is None:
            raise RuntimeError(
                'request handler must be used with xcube server'
            )
        self._context = RequestContext(server_context, request)

    @property
    def server_config(self) -> Mapping[str, Any]:
        return self._context.server_config

    @property
    def context(self) -> RequestContext:
        return self._context
