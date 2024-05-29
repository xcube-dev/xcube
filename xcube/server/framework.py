# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import abc
from typing import List, Type, Optional, Tuple
from collections.abc import Sequence

from xcube.constants import EXTENSION_POINT_SERVER_FRAMEWORKS
from xcube.util.extension import get_extension_registry
from xcube.util.jsonschema import JsonObjectSchema
from .api import ApiRoute
from .api import ApiStaticRoute
from .api import Context
from .asyncexec import AsyncExecution


class Framework(AsyncExecution, abc.ABC):
    """An abstract web server framework."""

    @property
    @abc.abstractmethod
    def config_schema(self) -> Optional[JsonObjectSchema]:
        """Returns an optional JSON Schema for
        the web server configuration. Returning None
        indicates that configuration is not possible.
        """

    @abc.abstractmethod
    def add_static_routes(self, routes: Sequence[ApiStaticRoute], url_prefix: str):
        """Adds the given static routes to this web server.

        Args:
            routes: The static routes to be added.
            url_prefix: URL prefix, may be an empty string.
        """

    @abc.abstractmethod
    def add_routes(self, routes: Sequence[ApiRoute], url_prefix: str):
        """Adds the given routes to this web server.

        Args:
            routes: The routes to be added.
            url_prefix: URL prefix, may be an empty string.
        """

    @abc.abstractmethod
    def update(self, ctx: Context):
        """Called, when the server context has changed.

        This is the case immediately after instantiation and before
        start() is called. It may then be called on any context change,
        likely due to a configuration change.

        Args:
            ctx: The current server context.
        """

    @abc.abstractmethod
    def start(self, ctx: Context):
        """Starts the web service.

        Args:
            ctx: The initial server context.
        """

    @abc.abstractmethod
    def stop(self, ctx: Context):
        """Stops the web service.

        Args:
            ctx: The current server context.
        """


def get_framework_names() -> list[str]:
    """Get the names of possible web server frameworks."""
    extension_registry = get_extension_registry()
    return [
        ext.name
        for ext in extension_registry.find_extensions(EXTENSION_POINT_SERVER_FRAMEWORKS)
    ]


def get_framework_class(framework_name: str) -> type[Framework]:
    """Get the web server framework class for the given *framework_name*."""
    extension_registry = get_extension_registry()
    return extension_registry.get_component(
        EXTENSION_POINT_SERVER_FRAMEWORKS, framework_name
    )
