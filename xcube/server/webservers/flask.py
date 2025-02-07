# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import concurrent.futures
from collections.abc import Awaitable, Sequence
from typing import Any, Callable, Optional, Tuple, Union

from xcube.server.api import ApiRoute, ApiStaticRoute, Context, ReturnT
from xcube.server.framework import Framework
from xcube.util.jsonschema import JsonObjectSchema


class FlaskFramework(Framework):
    """The Flask web server framework.

    TODO: implement me!
    """

    @property
    def config_schema(self) -> Optional[JsonObjectSchema]:
        return None

    def add_static_routes(
        self, static_routes: Sequence[ApiStaticRoute], url_prefix: str
    ):
        raise NotImplementedError()

    def add_routes(self, routes: Sequence[ApiRoute], url_prefix: str):
        raise NotImplementedError()

    def update(self, ctx: Context):
        raise NotImplementedError()

    def start(self, ctx: Context):
        raise NotImplementedError()

    def stop(self, ctx: Context):
        raise NotImplementedError()

    def call_later(
        self, delay: Union[int, float], callback: Callable, *args, **kwargs
    ) -> object:
        raise NotImplementedError()

    def run_in_executor(
        self,
        executor: Optional[concurrent.futures.Executor],
        function: Callable[..., ReturnT],
        *args: Any,
        **kwargs: Any,
    ) -> Awaitable[ReturnT]:
        raise NotImplementedError()
