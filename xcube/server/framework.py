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

import abc
from typing import Sequence, List, Type, Union, Callable, Optional, Any, \
    Awaitable, TypeVar

from tornado import concurrent

from xcube.constants import EXTENSION_POINT_SERVER_FRAMEWORKS
from xcube.util.extension import get_extension_registry
from .api import ApiRoute
from .api import Context
from .asyncexec import AsyncExecution

ReturnT = TypeVar("ReturnT")


class ServerFramework(AsyncExecution, abc.ABC):
    """
    An abstract web server framework.
    """

    @abc.abstractmethod
    def add_routes(self, routes: Sequence[ApiRoute]):
        """Adds the given routes to this web server."""

    @abc.abstractmethod
    def update(self, ctx: Context):
        """
        Called, when the server's root context has changed.

        This is the case immediately after instantiation and before
        start() is called. It may then be called on any context change,
        likely due to a configuration change.

        :param ctx: The current server's root context.
        """

    @abc.abstractmethod
    def start(self, ctx: Context):
        """
        Starts the web service.
        :param ctx: The initial server's root context.
        """

    @abc.abstractmethod
    def stop(self, ctx: Context):
        """
        Stops the web service.
        :param ctx: The current server's root context.
        """

    @abc.abstractmethod
    def call_later(self,
                   delay: Union[int, float],
                   callback: Callable,
                   *args,
                   **kwargs) -> object:
        """
        Executes the given callable *callback* after *delay* seconds.

        The method returns a handle that can be used to cancel the
        callback.

        :param delay: Delay in seconds.
        :param callback: Callback to be called.
        :param args: Positional arguments passed to *callback*.
        :param kwargs: Keyword arguments passed to *callback*.
        :return: A handle that provides the methods
            ``cancel()`` and ``cancelled()``.
        """

    @abc.abstractmethod
    def run_in_executor(
            self,
            executor: Optional[concurrent.futures.Executor],
            function: Callable[..., ReturnT],
            *args: Any,
            **kwargs: Any
    ) -> Awaitable[ReturnT]:
        """
        Concurrently runs a *function* in a ``concurrent.futures.Executor``.
        If *executor* is ``None``, the framework's default
        executor will be used.

        :param executor: An optional executor.
        :param function: The function to be run concurrently.
        :param args: Positional arguments passed to *function*.
        :param kwargs: Keyword arguments passed to *function*.
        :return: The awaitable return value of *function*.
        """


def get_framework_names() -> List[str]:
    """Get the names of possible web server frameworks."""
    extension_registry = get_extension_registry()
    return [
        ext.name for ext in extension_registry.find_extensions(
            EXTENSION_POINT_SERVER_FRAMEWORKS
        )
    ]


def get_framework_class(framework_name: str) -> Type[ServerFramework]:
    """Get the web server framework class for the given *framework_name*."""
    extension_registry = get_extension_registry()
    return extension_registry.get_component(
        EXTENSION_POINT_SERVER_FRAMEWORKS,
        framework_name
    )
