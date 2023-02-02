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
from typing import Sequence, List, Type, Optional, Tuple

from xcube.constants import EXTENSION_POINT_SERVER_FRAMEWORKS
from xcube.util.extension import get_extension_registry
from xcube.util.jsonschema import JsonObjectSchema
from .api import ApiRoute
from .api import ApiStaticRoute
from .api import Context
from .asyncexec import AsyncExecution


class Framework(AsyncExecution, abc.ABC):
    """
    An abstract web server framework.
    """

    @property
    @abc.abstractmethod
    def config_schema(self) -> Optional[JsonObjectSchema]:
        """Returns an optional JSON Schema for
        the web server configuration. Returning None
        indicates that configuration is not possible."""

    @abc.abstractmethod
    def add_static_routes(self,
                          routes: Sequence[ApiStaticRoute],
                          url_prefix: str):
        """
        Adds the given static routes to this web server.

        :param routes: The static routes to be added.
        :param url_prefix: URL prefix, may be an empty string.
        """

    @abc.abstractmethod
    def add_routes(self,
                   routes: Sequence[ApiRoute],
                   url_prefix: str):
        """
        Adds the given routes to this web server.

        :param routes: The routes to be added.
        :param url_prefix: URL prefix, may be an empty string.
        """

    @abc.abstractmethod
    def update(self, ctx: Context):
        """
        Called, when the server context has changed.

        This is the case immediately after instantiation and before
        start() is called. It may then be called on any context change,
        likely due to a configuration change.

        :param ctx: The current server context.
        """

    @abc.abstractmethod
    def start(self, ctx: Context):
        """
        Starts the web service.
        :param ctx: The initial server context.
        """

    @abc.abstractmethod
    def stop(self, ctx: Context):
        """
        Stops the web service.
        :param ctx: The current server context.
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
