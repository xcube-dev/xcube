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
from typing import Sequence, List, Type

from xcube.constants import EXTENSION_POINT_SERVER_FRAMEWORKS
from xcube.util.extension import get_extension_registry
from .api import ApiRoute
from .api import Context
from .asyncexec import AsyncExecution


class Framework(AsyncExecution, abc.ABC):
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


def get_framework_names() -> List[str]:
    """Get the names of possible web server frameworks."""
    extension_registry = get_extension_registry()
    return [
        ext.name for ext in extension_registry.find_extensions(
            EXTENSION_POINT_SERVER_FRAMEWORKS
        )
    ]


def get_framework_class(framework_name: str) -> Type[Framework]:
    """Get the web server framework class for the given *framework_name*."""
    extension_registry = get_extension_registry()
    return extension_registry.get_component(
        EXTENSION_POINT_SERVER_FRAMEWORKS,
        framework_name
    )
