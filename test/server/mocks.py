# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import (
    Optional,
    Tuple,
    Dict,
    Any,
    Union,
    Callable,
)
from collections.abc import Sequence, Awaitable, Mapping

from tornado import concurrent

from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.server.api import Api
from xcube.server.api import ApiContext
from xcube.server.api import ApiRequest
from xcube.server.api import ApiResponse
from xcube.server.api import ApiRoute
from xcube.server.api import ApiStaticRoute
from xcube.server.api import Context
from xcube.server.api import JSON
from xcube.server.api import ReturnT
from xcube.server.framework import Framework
from xcube.server.server import Server
from xcube.util.extension import ExtensionRegistry

ApiSpec = Union[Api, str, tuple[str, dict[str, Any]]]

ApiSpecs = Sequence[ApiSpec]


def mock_server(
    framework: Optional[Framework] = None,
    config: Optional[Mapping[str, Any]] = None,
    api_specs: Optional[ApiSpecs] = None,
) -> Server:
    return Server(
        framework or MockFramework(),
        config or {},
        extension_registry=mock_extension_registry(api_specs or ()),
    )


def mock_extension_registry(api_specs: ApiSpecs) -> ExtensionRegistry:
    extension_registry = ExtensionRegistry()
    for api in api_specs:
        if isinstance(api, str):
            api_name = api
            api = Api(api_name)
        elif not isinstance(api, Api):
            api_name, api_kwargs = api
            api = Api(api_name, **api_kwargs)
        extension_registry.add_extension(
            EXTENSION_POINT_SERVER_APIS, api.name, component=api
        )
    return extension_registry


class MockFramework(Framework):
    def __init__(self):
        self.add_static_routes_count = 0
        self.add_routes_count = 0
        self.update_count = 0
        self.start_count = 0
        self.stop_count = 0
        self.call_later_count = 0
        self.run_in_executor_count = 0

    @property
    def config_schema(self):
        return None

    def add_static_routes(
        self, static_routes: Sequence[ApiStaticRoute], url_prefix: str
    ):
        self.add_static_routes_count += 1

    def add_routes(self, routes: Sequence[ApiRoute], url_prefix: str):
        self.add_routes_count += 1

    def update(self, ctx: Context):
        self.update_count += 1

    def start(self, ctx: Context):
        self.start_count += 1

    def stop(self, ctx: Context):
        self.stop_count += 1

    def call_later(
        self, delay: Union[int, float], callback: Callable, *args, **kwargs
    ) -> object:
        self.call_later_count += 1
        return object()

    def run_in_executor(
        self,
        executor: Optional[concurrent.futures.Executor],
        function: Callable[..., ReturnT],
        *args: Any,
        **kwargs: Any,
    ) -> Awaitable[ReturnT]:
        self.run_in_executor_count += 1
        import concurrent.futures

        # noinspection PyTypeChecker
        return concurrent.futures.Future()


class MockApiContext(ApiContext):
    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self.on_update_count = 0
        self.on_dispose_count = 0

    def on_update(self, prev_ctx: Optional[ApiContext]):
        self.on_update_count += 1

    def on_dispose(self):
        self.on_dispose_count += 1


class MockApiRequest(ApiRequest):
    def __init__(
        self,
        query_args: Optional[Mapping[str, Sequence[str]]] = None,
        reverse_url_prefix: str = "",
    ):
        self._base_url = "http://localhost:8080"
        self._query_args = query_args or {}
        self._reverse_url_prefix = reverse_url_prefix

    @property
    def query(self) -> Mapping[str, Sequence[str]]:
        return self._query_args

    # noinspection PyShadowingBuiltins
    def get_query_args(self, name: str, type: Any = None) -> Sequence[Any]:
        args = self._query_args.get(name, [])
        return [type(arg) for arg in args] if type is not None else args

    def url_for_path(
        self, path: str, query: Optional[str] = None, reverse: bool = False
    ) -> str:
        if path and not path.startswith("/"):
            path = "/" + path
        prefix = self._reverse_url_prefix if reverse else ""
        return f"{self._base_url}{prefix}{path}" + (f"?{query}" if query else "")

    @property
    def headers(self) -> Mapping[str, str]:
        return {}

    @property
    def url(self) -> str:
        query = "&".join(
            [
                "&".join([f"{key}={value}" for value in values])
                for key, values in self.query.items()
            ]
        )
        return self.url_for_path("datasets", query=query)

    @property
    def body(self) -> bytes:
        return bytes('{"dataset": []}')

    @property
    def json(self) -> JSON:
        return dict(datasets=[])

    def make_query_lower_case(self):
        pass


class MockApiResponse(ApiResponse):
    def set_header(self, name: str, value: str):
        pass

    def set_status(self, status_code: int, reason: Optional[str] = None):
        pass

    def write(self, data: Union[str, bytes, JSON], content_type: Optional[str] = None):
        pass

    def finish(
        self, data: Union[str, bytes, JSON] = None, content_type: Optional[str] = None
    ):
        pass
