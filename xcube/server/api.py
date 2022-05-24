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
from typing import Any, List, Optional, Tuple, Dict, Union, Type, Sequence, \
    Generic, TypeVar

import tornado.httputil
import tornado.ioloop
import tornado.web

from .config import Config
from .context import Context
from ..util.jsonschema import JsonObjectSchema

_SERVER_CONTEXT_ATTR_NAME = '__xcube_server_context'

ApiRoute = Union[
    Tuple[str, Type["ApiHandler"]],
    Tuple[str, Type["ApiHandler"], Dict[str, Any]]
]

# API Context type variable
ApiContextT = TypeVar("ApiContextT", bound="ApiContext")


class Api(Generic[ApiContextT]):
    """
    A server API.

    The most common purpose of this class is to
    add a new API to the server by the means of routes.

    Every may produce API context objects for a given server
    configuration.

    If the server configuration changes, the API is asked to
    create a new context object.

    However, an API may be just programmatic and serve as a
    web server middleware. It can then still provide
    the context for other dependent APIs.

    May be derived by clients to override the methods

    * `on_start`,
    * `on_stop`,
    * `create_ctx`.

    Each extension API module must export an instance of this
    class. A typical use case of this class:

    ```
    class DatasetsApiContext(ApiContext)
        def update(self, prev_ctx: Optional[Context]):
            config = self.config
            ...
        def get_datasets(self):
            ...

    class DatasetsApi(Api[DatasetsApiContext]):
        def __init__(self):
            super().__init__("datasets",
                             config_schema=DATASET_CONFIG_SCHEMA)

        def create_ctx(self, root_ctx: Context):
            return DatasetsApiContext(root_ctx)

    api = DatasetsApi()

    @api.route("/datasets")
    class DatasetsHandler(ApiHandler[DatasetsApiContext]):
        def get(self):
            return self.ctx.get_datasets()
    ```

    :param routes: Optional list of initial routes.
        A route is a tuple of the form (route-pattern, handler-class) or
        (route-pattern, handler-class, handler-kwargs). The handler-class
        must be derived from ApiHandler.
    :param required_apis: Sequence of names of other required APIs.
    :param optional_apis: Sequence of names of other optional APIs.
    :param config_schema: Optional JSON schema for the API's configuration.
        If not given, or None is passed, the API is assumed to
        have no configuration.
    :param api_ctx_cls: Optional API context class.
        If given, it must be derived from ApiContext.
    """

    def __init__(self,
                 name: str, /,
                 routes: Optional[Sequence[ApiRoute]] = None,
                 required_apis: Optional[Sequence[str]] = None,
                 optional_apis: Optional[Sequence[str]] = None,
                 config_schema: Optional[JsonObjectSchema] = None,
                 api_ctx_cls: Optional[Type[ApiContextT]] = None):
        self._name = name
        self._required_apis = tuple(required_apis or ())
        self._optional_apis = tuple(optional_apis or ())
        self._routes: List[ApiRoute] = list(routes or [])
        self._config_schema = config_schema
        self._api_ctx_cls = api_ctx_cls

    @property
    def name(self) -> str:
        """The name of this API."""
        return self._name

    @property
    def required_apis(self) -> Tuple[str]:
        """The names of other required APIs."""
        return self._required_apis

    @property
    def optional_apis(self) -> Tuple[str]:
        """The names of other optional APIs."""
        return self._required_apis

    def route(self, pattern: str, **target_kwargs):
        """
        Decorator that adds a route to this API.

        The decorator target must be a class derived from ApiHandler.

        :param pattern: The route pattern.
        :param target_kwargs: Optional keyword arguments passed to
            RequestHandler constructor.
        :return: A decorator function that receives a
            class derived from RequestHandler
        """

        def decorator_func(target_class: Type["ApiHandler"]):
            if not issubclass(target_class, ApiHandler):
                raise TypeError(f'target_class must be an'
                                f' instance of {ApiHandler.__name__},'
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
    def config_schema(self) -> Optional[JsonObjectSchema]:
        """
        Get the JSON schema for the configuration of this API.
        """
        return self._config_schema

    def on_start(self,
                 root_ctx: Context,
                 io_loop: tornado.ioloop.IOLoop):
        """
        Called when the server is started.

        :param root_ctx: The current root context
        :param io_loop: The current i/o loop.
        """

    def on_stop(self,
                root_ctx: Context,
                io_loop: tornado.ioloop.IOLoop):
        """
        Called when the server is stopped.

        :param root_ctx: The current root context
        :param io_loop: The current i/o loop.
        """

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def create_ctx(self, root_ctx: Context) -> Optional[ApiContextT]:
        """
        Create a new context object for this API.
        If the API doesn't require a context object, the method should
        return None.
        The default implementation uses the *api_ctx_cls* class, if any,
        to instantiate an API context using root_ctx as only argument.
        Otherwise, None is returned.

        :param root_ctx: The root context.
        :return: An instance of ApiContext or None
        """
        if self._api_ctx_cls is not None:
            assert issubclass(self._api_ctx_cls, ApiContext)
            return self._api_ctx_cls(root_ctx)
        return None


class ApiContext(Context, abc.ABC):
    """
    An abstract base class for API context objects.

    A typical use case is to cache computationally expensive
    resources served by a particular API.

    Derived classes

    * must implement the `update()` method in order
      to initialize or update this context object state with
      respect to the current server configuration, or with
      respect to other API context object states.
    * may overwrite the `dispose()` method to empty any caches
      and close access to resources.
    * must call the super class constructor with the *root* context,
      from their own constructor, if any.

    :param root: The root (server) context object.
    """

    def __init__(self, root: Context):
        self._root = root

    @property
    def root(self) -> Context:
        """The root (server) context object."""
        return self._root

    @property
    def config(self) -> Config:
        return self.root.config

    def get_api_ctx(self, api_name: str) -> Optional["ApiContext"]:
        return self.root.get_api_ctx(api_name)

    def dispose(self):
        """Does nothing."""


class ApiHandler(tornado.web.RequestHandler,
                 Generic[ApiContextT],
                 abc.ABC):
    """
    Base class for all API handlers.

    :param application: The Tornado Application
    :param request: The current request
    :param kwargs: Parameters passed to the handler.
    """

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

        root_ctx = getattr(application,
                           _SERVER_CONTEXT_ATTR_NAME, None)
        from .server import ServerContext
        if not isinstance(root_ctx, ServerContext):
            raise RuntimeError(
                'request handler must be used with xcube server'
            )

        self._root_ctx = root_ctx
        self._ctx = root_ctx.get_api_ctx(self.api_name)

    @property
    def config(self) -> Config:
        """The server configuration."""
        return self._root_ctx.config

    @property
    def root_ctx(self) -> Context:
        """The root (server) context."""
        return self._root_ctx

    @property
    def ctx(self) -> Optional[ApiContextT]:
        """The API's context object, or None, if not defined."""
        # noinspection PyTypeChecker
        return self._ctx

