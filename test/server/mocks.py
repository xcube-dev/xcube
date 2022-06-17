#  The MIT License (MIT)
#  Copyright (c) 2022 by the xcube development team and contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

from typing import Optional, Sequence, Tuple, Dict, Any, Union, Callable, \
    Awaitable, List, Mapping

from tornado import concurrent

from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.server.api import Api
from xcube.server.api import ApiContext
from xcube.server.api import ApiRequest
from xcube.server.api import ApiResponse
from xcube.server.api import ApiRoute
from xcube.server.api import Context
from xcube.server.api import JSON
from xcube.server.api import ReturnT
from xcube.server.api import ServerConfig
from xcube.server.framework import Framework
from xcube.server.server import Server
from xcube.util.extension import ExtensionRegistry

ApiSpec = Union[Api,
                str,
                Tuple[str, Dict[str, Any]]]

ApiSpecs = Sequence[ApiSpec]


def mock_server(framework: Optional[Framework] = None,
                config: Optional[ServerConfig] = None,
                api_specs: Optional[ApiSpecs] = None) -> Server:
    return Server(
        framework or MockFramework(),
        config or {},
        extension_registry=mock_extension_registry(api_specs or ())
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
        extension_registry.add_extension(EXTENSION_POINT_SERVER_APIS,
                                         api.name,
                                         component=api)
    return extension_registry


class MockFramework(Framework):

    def __init__(self):
        self.add_routes_count = 0
        self.update_count = 0
        self.start_count = 0
        self.stop_count = 0
        self.call_later_count = 0
        self.run_in_executor_count = 0

    def add_routes(self, routes: Sequence[ApiRoute]):
        self.add_routes_count += 1

    def update(self, ctx: Context):
        self.update_count += 1

    def start(self, ctx: Context):
        self.start_count += 1

    def stop(self, ctx: Context):
        self.stop_count += 1

    def call_later(self,
                   delay: Union[int, float],
                   callback: Callable,
                   *args,
                   **kwargs) -> object:
        self.call_later_count += 1
        return object()

    def run_in_executor(self,
                        executor: Optional[concurrent.futures.Executor],
                        function: Callable[..., ReturnT],
                        *args: Any,
                        **kwargs: Any) -> Awaitable[ReturnT]:
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

    def __init__(self,
                 query_args: Optional[Mapping[str, Sequence[str]]] = None):
        self._query_args = query_args or {}

    @property
    def query(self) -> Mapping[str, Sequence[str]]:
        return self._query_args

    # noinspection PyShadowingBuiltins
    def get_query_args(self, name: str, type: Any = None) -> Sequence[Any]:
        args = self._query_args.get(name, [])
        return [type(arg) for arg in args] if type is not None else args

    def url_for_path(self, path: str, query: Optional[str] = None) -> str:
        return ''

    @property
    def headers(self) -> Mapping[str, str]:
        return {}

    @property
    def url(self) -> str:
        return ''

    @property
    def body(self) -> bytes:
        return bytes({})

    @property
    def json(self) -> JSON:
        return {}



class MockApiResponse(ApiResponse):
    def set_header(self, name: str, value: str):
        pass

    def set_status(self, status_code: int, reason: Optional[str] = None):
        pass

    def write(self, data: Union[str, bytes, JSON]):
        pass

    def finish(self, data: Union[str, bytes, JSON] = None):
        pass
