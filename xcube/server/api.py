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
from typing import Any, List, Optional, Tuple, Dict, Union, Type, Sequence, Generic, TypeVar

import tornado.httputil
import tornado.ioloop
import tornado.web

from .config import ServerConfig
from .context import Context
from .context import ServerContext
from ..util.jsonschema import JsonSchema

SERVER_CONTEXT_ATTR_NAME = '__xcube_server_context'

ApiRoute = Union[
    Tuple[str, Type["ApiHandler"]],
    Tuple[str, Type["ApiHandler"], Dict[str, Any]]
]

# ConteXt type variable
X = TypeVar("X", bound="Api")


class Api(Generic[X]):
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
                 routes: Optional[Sequence[ApiRoute]] = None,
                 config_schema: Optional[JsonSchema] = None):
        self._name = name
        self._dependencies = tuple(dependencies or ())
        self._routes: List[ApiRoute] = list(routes or [])
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

        def decorator_func(target_class: Type["ApiHandler"]):
            if not issubclass(target_class, ApiHandler):
                raise TypeError(f'target_class must be an'
                                f' instance of {ApiHandler},'
                                f' but was {target_class}')
            if target_kwargs:
                handler = pattern, target_class, target_kwargs
            else:
                handler = pattern, target_class
            self._routes.append(handler)

        return decorator_func

    @property
    def routes(self) -> List[ApiRoute]:
        """The routes provided by this API."""
        return self._routes

    @property
    def config_schema(self) -> Optional[JsonSchema]:
        """
        Get the JSON schema for the configuration of this API.
        """
        return self._config_schema

    def on_start(self,
                 server_context: Context,
                 io_loop: tornado.ioloop.IOLoop):
        """
        Called when the server is started.

        :param server_context: The current server context
        :param io_loop: The current i/o loop.
        """

    def on_stop(self,
                server_context: Context,
                io_loop: tornado.ioloop.IOLoop):
        """
        Called when the server is stopped.

        :param server_context: The current server context
        :param io_loop: The current i/o loop.
        """

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def create_context(self, server_context: ServerContext) -> Optional[X]:
        """
        Create a new context object for this API.
        If the API doesn't require a context object, None is returned.
        The default implementation returns None.

        :param server_context: The server context.
        :return: An instance of ApiContext or None
        """
        return None


class ApiHandler(tornado.web.RequestHandler, Generic[X], abc.ABC):
    api_name: str

    def __init__(self,
                 application: tornado.web.Application,
                 request: tornado.httputil.HTTPServerRequest,
                 **kwargs: Any):
        super().__init__(application, request, **kwargs)
        if not isinstance(self.api_name, str):
            raise RuntimeError(
                'request handler must be used with xcube server'
            )

        server_context = getattr(application,
                                 SERVER_CONTEXT_ATTR_NAME, None)
        if server_context is None:
            raise RuntimeError(
                'request handler must be used with xcube server'
            )

        self._server_context = server_context
        self._api_context = server_context.get_api_context(self.api_name)

    @property
    def server_config(self) -> ServerConfig:
        return self._server_context.server_config

    @property
    def server_context(self) -> ServerContext:
        return self._server_context

    @property
    def api_context(self) -> X:
        return self._api_context
