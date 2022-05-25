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

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Dict, Type, Sequence, \
    Generic, TypeVar, Union

from .config import Config
from .context import Context
from ..util.assertions import assert_instance, assert_true
from ..util.jsonschema import JsonObjectSchema

_SERVER_CONTEXT_ATTR_NAME = '__xcube_server_context'

# API Context type variable
ApiContextT = TypeVar("ApiContextT", bound="ApiContext")

JSON = Union[
    None,
    bool,
    int,
    float,
    str,
    List["JSON"],
    Dict[str, "JSON"],
]


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

    * `start` - to do things on server start;
    * `stop` - to do things on server stop;
    * `create_ctx` - to create an API-specific context object.

    Each extension API module must export an instance of this
    class. A typical use case of this class:

    ```
    class DatasetsContext(ApiContext)
        def update(self, prev_ctx: Optional[Context]):
            config = self.config
            ...

        def get_datasets(self):
            ...

    class DatasetsApi(Api[DatasetsContext]):
        def __init__(self):
            super().__init__("datasets",
                             config_schema=DATASET_CONFIG_SCHEMA)

        def create_ctx(self, root_ctx: Context):
            return DatasetsApiContext(root_ctx)

    api = DatasetsApi()

    @api.route("/datasets")
    class DatasetsHandler(ApiHandler[DatasetsContext]):
        def get(self):
            return self.ctx.get_datasets()
    ```

    :param name: The API name. Must be unique within a server.
    :param version: The API version. Defaults to "0.0.0".
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
                 version: str = '0.0.0',
                 routes: Optional[Sequence["ApiRoute"]] = None,
                 required_apis: Optional[Sequence[str]] = None,
                 optional_apis: Optional[Sequence[str]] = None,
                 config_schema: Optional[JsonObjectSchema] = None,
                 api_ctx_cls: Optional[Type[ApiContextT]] = None):
        self._name = name
        self._version = version
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
    def version(self) -> str:
        """The version of this API."""
        return self._version

    @property
    def required_apis(self) -> Tuple[str]:
        """The names of other required APIs."""
        return self._required_apis

    @property
    def optional_apis(self) -> Tuple[str]:
        """The names of other optional APIs."""
        return self._required_apis

    def route(self, pattern: str, **handler_kwargs):
        """
        Decorator that adds a route to this API.

        The decorator target must be a class derived from ApiHandler.

        :param pattern: The route pattern.
        :param handler_kwargs: Optional keyword arguments passed to
            ApiHandler constructor.
        :return: A decorator function that receives a
            class derived from ApiHandler
        """

        def decorator_func(handler_cls: Type[ApiHandler]):
            self._routes.append(ApiRoute(self.name,
                                         pattern,
                                         handler_cls,
                                         handler_kwargs))

        return decorator_func

    @property
    def routes(self) -> List["ApiRoute"]:
        """The routes provided by this API."""
        return self._routes

    @property
    def config_schema(self) -> Optional[JsonObjectSchema]:
        """
        Get the JSON schema for the configuration of this API.
        """
        return self._config_schema

    def start(self, root_ctx: Context):
        """
        Called when the server is started.

        :param root_ctx: The server's current root context
        """

    def stop(self, root_ctx: Context):
        """
        Called when the server is stopped.

        :param root_ctx: The server's current root context
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

        :param root_ctx: The server's current root context.
        :return: An instance of ApiContext or None
        """
        if self._api_ctx_cls is not None:
            assert issubclass(self._api_ctx_cls, ApiContext)
            return self._api_ctx_cls(root_ctx)
        return None


class ApiContext(Context, ABC):
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


class ApiRequest:
    @property
    @abstractmethod
    def body(self) -> bytes:
        """The request body."""

    @property
    @abstractmethod
    def json(self) -> JSON:
        """The request body as JSON value."""

    def get_query_arg(self, name: str) -> Optional[str]:
        """Get the value of query argument given by *name*."""
        args = self.get_query_args(name)
        return args[0] if len(args) > 0 else None

    @abstractmethod
    def get_query_args(self, name: str) -> Sequence[str]:
        """Get the values of query argument given by *name*."""

    def get_body_arg(self, name: str) -> Optional[bytes]:
        args = self.get_body_args(name)
        return args[0] if len(args) > 0 else None

    @abstractmethod
    def get_body_args(self, name: str) -> Sequence[bytes]:
        """Get the values of body argument given by *name*."""


class ApiResponse(ABC):
    @abstractmethod
    def write(self, data: Union[str, bytes, JSON]):
        """Write data."""

    @abstractmethod
    def finish(self, data: Union[str, bytes, JSON] = None):
        """Finish the response (and submit it)."""

    @abstractmethod
    def error(self,
              status_code: int,
              message: Optional[str] = None,
              *args: Any,
              **kwargs: Any) -> Exception:
        """
        Get an exception that can be raised.
        If raised, a standard error response will be generated.
        """


class ApiHandler(Generic[ApiContextT], ABC):
    """
    Base class for all API handlers.

    :param api_name: The name of the API that defines this handler.
    :param root_ctx: The server's root context.
    :param request: The API handler's request.
    :param response: The API handler's response.
    :param kwargs: Client keyword arguments (not used in base class).
    """

    def __init__(self,
                 api_name: str,
                 root_ctx: Context,
                 request: ApiRequest,
                 response: ApiResponse,
                 **kwargs: Any):
        self._root_ctx = root_ctx
        self._ctx = root_ctx.get_api_ctx(api_name)
        self._request = request
        self._response = response
        self._kwargs = kwargs

    @property
    def request(self) -> ApiRequest:
        return self._request

    @property
    def response(self) -> ApiResponse:
        return self._response

    @property
    def config(self) -> Config:
        """The server configuration."""
        return self._root_ctx.config

    @property
    def root_ctx(self) -> Context:
        """The server's root context."""
        return self._root_ctx

    @property
    def ctx(self) -> Optional[ApiContextT]:
        """The API's context object, or None, if not defined."""
        # noinspection PyTypeChecker
        return self._ctx

    def _unimplemented_method(self, *args: str, **kwargs: str) -> None:
        raise self.response.error(405)

    get = _unimplemented_method
    post = _unimplemented_method
    put = _unimplemented_method
    delete = _unimplemented_method
    options = _unimplemented_method


class ApiRoute:
    def __init__(self,
                 api_name: str,
                 pattern: str,
                 handler_cls: Type[ApiHandler],
                 handler_kwargs: Optional[Dict[str, Any]] = None):
        assert_instance(api_name, str, name="api_name")
        assert_instance(pattern, str, name="pattern")
        assert_true(issubclass(handler_cls, ApiHandler),
                    message=f'handler_cls must be a subclass'
                            f' of {ApiHandler.__name__},'
                            f' was {handler_cls}')
        assert_instance(handler_kwargs, (type(None), dict),
                        name="handler_kwargs")
        self.api_name = api_name
        self.pattern = pattern
        self.handler_cls = handler_cls
        self.handler_kwargs = handler_kwargs
