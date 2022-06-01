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

import concurrent.futures
import inspect
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Dict, Type, Sequence, \
    Generic, TypeVar, Union, Callable, Awaitable, Mapping

from .asyncexec import AsyncExecution
from ..util.assertions import assert_instance
from ..util.assertions import assert_true
from ..util.jsonschema import JsonObjectSchema

_SERVER_CONTEXT_ATTR_NAME = '__xcube_server_context'
_HTTP_METHODS = {'get', 'post', 'put', 'delete', 'options'}

ArgT = TypeVar('ArgT')
ReturnT = TypeVar('ReturnT')
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

Config = Mapping[str, Any]

_builtin_type = type


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
    :param create_ctx: Optional API context factory.
        If given, must be a callable that accepts the server root context
        as only argument and returns an instance ApiContext or None.
    """

    def __init__(
            self,
            name: str, /,
            version: str = '0.0.0',
            description: Optional[str] = None,
            routes: Optional[Sequence["ApiRoute"]] = None,
            required_apis: Optional[Sequence[str]] = None,
            optional_apis: Optional[Sequence[str]] = None,
            config_schema: Optional[JsonObjectSchema] = None,
            create_ctx: Optional[
                Callable[["Context"], Optional[ApiContextT]]
            ] = None,
            on_start: Optional[
                Callable[["Context"], Any]
            ] = None,
            on_stop: Optional[
                Callable[["Context"], Any]
            ] = None,
    ):
        assert_instance(name, str, 'name')
        assert_instance(version, str, 'version')
        if description is not None:
            assert_instance(description, str, 'description')
        if config_schema is not None:
            assert_instance(config_schema, JsonObjectSchema, 'config_schema')
        if on_start is not None:
            assert_true(callable(on_start),
                        message='on_start must be callable')
        if on_stop is not None:
            assert_true(callable(on_stop),
                        message='on_stop must be callable')
        self._name = name
        self._version = version
        self._description = description
        self._required_apis = tuple(required_apis or ())
        self._optional_apis = tuple(optional_apis or ())
        self._routes: List[ApiRoute] = list(routes or [])
        self._config_schema = config_schema
        self._create_ctx = create_ctx
        self._on_start = on_start
        self._on_stop = on_stop

    @property
    def name(self) -> str:
        """The name of this API."""
        return self._name

    @property
    def version(self) -> str:
        """The version of this API."""
        return self._version

    @property
    def description(self) -> Optional[str]:
        """The description of this API."""
        return self._description or (getattr(self, '__doc__', None)
                                     if self.__class__ is not Api else None)

    @property
    def required_apis(self) -> Tuple[str]:
        """The names of other required APIs."""
        return self._required_apis

    @property
    def optional_apis(self) -> Tuple[str]:
        """The names of other optional APIs."""
        return self._required_apis

    def route(self, path: str, **handler_kwargs):
        """
        Decorator that adds a route to this API.

        The decorator target must be a class derived from ApiHandler.

        :param path: The route path.
        :param handler_kwargs: Optional keyword arguments passed to
            ApiHandler constructor.
        :return: A decorator function that receives a
            class derived from ApiHandler
        """

        def decorator_func(handler_cls: Type[ApiHandler]):
            self._routes.append(ApiRoute(self.name,
                                         path,
                                         handler_cls,
                                         handler_kwargs))
            return handler_cls

        return decorator_func

    def operation(self,
                  operation_id: Optional[str] = None,
                  summary: Optional[str] = None,
                  description: Optional[str] = None,
                  parameters: Optional[List[Dict[str, Any]]] = None,
                  **kwargs):
        """
        Decorator that adds OpenAPI 3.0 information to an
        API handler's operation,
        i.e. one of the get, post, put, delete, or options methods.

        :return: A decorator function that receives a
            and returns an ApiHandler's operation.
        """
        openapi = {
            "operationId": operation_id or kwargs.pop("operationId", None),
            "summary": summary,
            "description": description,
            "parameters": parameters,
        }
        openapi = {k: v for k, v in openapi.items() if v is not None}

        def decorator_func(target: Union[Type[ApiHandler], Callable]):
            if inspect.isfunction(target) \
                    and hasattr(target, '__name__') \
                    and target.__name__ in _HTTP_METHODS:
                setattr(target, "__openapi__", openapi)
            else:
                raise TypeError(f'API {self.name}:'
                                f' @operation() decorator'
                                f' must be used with one of the'
                                f' HTTP methods of an {ApiHandler.__name__}')
            return target

        return decorator_func

    @property
    def routes(self) -> Tuple["ApiRoute"]:
        """The routes provided by this API."""
        return tuple(self._routes)

    @property
    def config_schema(self) -> Optional[JsonObjectSchema]:
        """Get the JSON schema for the configuration of this API."""
        return self._config_schema

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def create_ctx(self, root_ctx: "Context") -> Optional[ApiContextT]:
        """Create a new context object for this API.
        If the API doesn't require a context object, the method should
        return None.
        The default implementation uses the *create_ctx*
        argument passed to the constructor, if any,
        to instantiate an API context using *root_ctx* as only argument.
        Otherwise, None is returned.
        Should not be called directly.

        :param root_ctx: The server's current root context.
        :return: An instance of ApiContext or None
        """
        if self._create_ctx is not None:
            return self._create_ctx(root_ctx)
        return None

    def on_start(self, root_ctx: "Context"):
        """Called when the server is started.
        Can be overridden to initialize the API.
        Should not be called directly.

        The default implementation calls the *on_start*
        argument passed to the constructor, if any.

        :param root_ctx: The server's current root context
        """
        if self._on_start is not None:
            return self._on_start(root_ctx)

    def on_stop(self, root_ctx: "Context"):
        """Called when the server is stopped.
        Can be overridden to initialize the API.
        Should not be called directly.

        The default implementation calls the *on_stop*
        argument passed to the constructor, if any.

        :param root_ctx: The server's current root context
        """
        if self._on_stop is not None:
            return self._on_stop(root_ctx)


class Context(AsyncExecution, ABC):
    """The interface for context objects."""

    @property
    @abstractmethod
    def apis(self) -> Tuple[Api]:
        """The APIs used by the server."""

    @property
    @abstractmethod
    def config(self) -> Config:
        """The server's current configuration."""

    @property
    @abstractmethod
    def root(self) -> "Context":
        """The server's current root context."""

    @abstractmethod
    def get_api_ctx(self, api_name: str) -> Optional["Context"]:
        """
        Get the API context for *api_name*.
        Can be used to access context objects of other APIs.

        :param api_name: The name of a registered API.
        :return: The API context for *api_name*, or None if no such exists.
        """

    @abstractmethod
    def on_update(self, prev_context: Optional["Context"]):
        """Called when the server configuration changed.
        Must be implemented by derived classes in order to update
        this context with respect to the current configuration
        ``self.config`` and the given *prev_context*, if any.
        The method shall not be called directly.

        :param prev_context: The previous context instance.
            Will be ``None`` if ``on_update()`` is called for the
            very first time.
        """

    @abstractmethod
    def on_dispose(self):
        """Called if this context will never be used again.
        May be overridden by derived classes in order to
        dispose allocated resources.
        The default implementation does nothing.
        The method shall not be called directly.
        """


class ApiContext(Context, ABC):
    """
    An abstract base class for API context objects.

    A typical use case is to cache computationally expensive
    resources served by a particular API.

    Derived classes

    * must implement the `on_update()` method in order
      to initialize or update this context object state with
      respect to the current server configuration, or with
      respect to other API context object states.
    * may overwrite the `on_dispose()` method to empty any caches
      and close access to resources.
    * must call the super class constructor with the *root* context,
      from their own constructor, if any.

    :param root: The server's root context.
    """

    def __init__(self, root: Context):
        self._root = root

    @property
    def root(self) -> Context:
        """The server context object."""
        return self._root

    @property
    def apis(self) -> Tuple[Api]:
        return self.root.apis

    @property
    def config(self) -> Config:
        return self.root.config

    def get_api_ctx(self, api_name: str) -> Optional["ApiContext"]:
        return self.root.get_api_ctx(api_name)

    def on_dispose(self):
        """Does nothing."""

    def call_later(self,
                   delay: Union[int, float],
                   callback: Callable,
                   *args,
                   **kwargs) -> object:
        return self.root.call_later(delay, callback,
                                    *args, **kwargs)

    def run_in_executor(self,
                        executor: Optional[concurrent.futures.Executor],
                        function: Callable[..., ReturnT],
                        *args: Any,
                        **kwargs: Any) -> Awaitable[ReturnT]:
        return self.root.run_in_executor(executor, function,
                                         *args, **kwargs)


class ApiRequest:
    @property
    @abstractmethod
    def body(self) -> bytes:
        """The request body."""

    @property
    @abstractmethod
    def json(self) -> JSON:
        """The request body as JSON value."""

    # noinspection PyShadowingBuiltins
    def get_query_arg(self, name: str,
                      type: Optional[Type[ArgT]] = None,
                      default: Any = None) -> Optional[ArgT]:
        """Get the value of query argument given by *name*."""
        if type is None and default is not None:
            type = _builtin_type(default)
            type = type if callable(type) else None
        values = self.get_query_args(name, type=type)
        return values[0] if values else default

    # noinspection PyShadowingBuiltins
    @abstractmethod
    def get_query_args(self,
                       name: str,
                       type: Optional[Type[ArgT]] = None) -> Sequence[ArgT]:
        """Get the values of query argument given by *name*."""

    def get_body_arg(self, name: str) -> Optional[bytes]:
        args = self.get_body_args(name)
        return args[0] if len(args) > 0 else None

    @abstractmethod
    def get_body_args(self, name: str) -> Sequence[bytes]:
        """Get the values of body argument given by *name*."""


class ApiResponse(ABC):
    @abstractmethod
    def set_status(self, status_code: int, reason: Optional[str] = None):
        """Set the HTTP status code and optionally the reason."""

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
              reason: Optional[str] = None) -> Exception:
        """
        Get an exception that can be raised.
        If raised, a standard error response will be generated.

        :param status_code: The HTTP status code.
        :param message: Optional message.
        :param reason: Optional reason.
        :return: An exception that may be raised.
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
        """The request that provides the handler's input."""
        return self._request

    @property
    def response(self) -> ApiResponse:
        """The response that provides the handler's output."""
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
                 path: str,
                 handler_cls: Type[ApiHandler],
                 handler_kwargs: Optional[Dict[str, Any]] = None):
        assert_instance(api_name, str, name="api_name")
        assert_instance(path, str, name="path")
        assert_instance(handler_cls, type, name="handler_cls")
        assert_true(issubclass(handler_cls, ApiHandler),
                    message=f'handler_cls must be a subclass'
                            f' of {ApiHandler.__name__},'
                            f' was {handler_cls}')
        assert_instance(handler_kwargs, (type(None), dict),
                        name="handler_kwargs")
        self.api_name = api_name
        self.path = path
        self.handler_cls = handler_cls
        self.handler_kwargs = dict(handler_kwargs or {})

    def __eq__(self, other) -> bool:
        if isinstance(other, ApiRoute):
            return self.api_name == other.api_name \
                   and self.path == other.path \
                   and self.handler_cls == other.handler_cls \
                   and self.handler_kwargs == other.handler_kwargs
        return False

    def __hash__(self) -> int:
        return hash(self.api_name) \
               + 2 * hash(self.path) \
               + 4 * hash(self.handler_cls) \
               + 16 * hash(tuple(sorted(tuple(self.handler_kwargs.items()),
                                        key=lambda p: p[0])))

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        args = (f"{self.api_name!r},"
                f" {self.path!r},"
                f" {self.handler_cls.__name__}")
        if self.handler_kwargs:
            args += f", handler_kwargs={self.handler_kwargs!r}"
        return f"ApiRoute({args})"
