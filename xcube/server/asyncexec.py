# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import abc
from typing import Union, Callable, Optional, Any, TypeVar
from collections.abc import Awaitable

from tornado import concurrent

ReturnT = TypeVar("ReturnT")


class AsyncExecution(abc.ABC):
    """
    An interface that defines asynchronous function execution.
    """

    @abc.abstractmethod
    def call_later(
        self, delay: Union[int, float], callback: Callable, *args, **kwargs
    ) -> object:
        """Executes the given callable *callback* after *delay* seconds.

        The method returns a handle that can be used to cancel the
        callback.

        Args:
            delay: Delay in seconds.
            callback: Callback to be called.
            args: Positional arguments passed to *callback*.
            kwargs: Keyword arguments passed to *callback*.

        Returns: A handle that provides the methods
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
        """Concurrently runs a *function* in a
        ``concurrent.futures.Executor``.
        If *executor* is ``None``, the framework's default
        executor will be used.

        Args:
            executor: An optional executor.
            function: The function to be run concurrently.
            args: Positional arguments passed to *function*.
            kwargs: Keyword arguments passed to *function*.

        Returns: The awaitable return value of *function*.
        """
