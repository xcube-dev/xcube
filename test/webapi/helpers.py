# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import os
from typing import Any
from collections.abc import Mapping
from typing import Optional, Type, TypeVar
from typing import Union

import yaml

from test.server.mocks import MockFramework
from xcube.server.api import Context
from xcube.server.framework import Framework
from xcube.server.server import Server
from xcube.server.testing import ServerTestCase
from xcube.util.extension import ExtensionRegistry
from xcube.util.plugin import get_extension_registry

XCUBE_TEST_CLIENT_ID = os.environ.get("XCUBE_TEST_CLIENT_ID")
XCUBE_TEST_CLIENT_SECRET = os.environ.get("XCUBE_TEST_CLIENT_SECRET")

T = TypeVar("T", bound=Context)


def get_server(
    server_config: Optional[Union[str, Mapping[str, Any]]] = None,
    framework: Optional[Framework] = None,
    extension_registry: Optional[ExtensionRegistry] = None,
) -> Server:
    """Get the server object for the given
    server configuration path or dictionary *server_config*.

    This function is used for testing API contexts and controllers.

    Args:
        server_config: Server configuration string or mapping.
            If it is just a filename, it is resolved against test resource
            directory "${project}/test/webapi/res".
            If it is a relative path, it is resolved against the current
            working directory (not recommended).
            Defaults to ``'config.yml'``.
        framework: Web framework, defaults to a MockFramework.
        extension_registry: Extension registry,
            defaults to xcube's populated default extension registry

    Returns:
        The API context object

    Raise:
        AssertionError: if API context object can not be determined
    """
    server_config = get_server_config(server_config)
    framework = framework or MockFramework()
    extension_registry = extension_registry or get_extension_registry()
    return Server(framework, server_config, extension_registry=extension_registry)


def get_server_config(
    server_config: Optional[Union[str, Mapping[str, Any]]] = None
) -> dict[str, Any]:
    """Get a server configuration for testing.

    The given ``server_config`` is normalized into a dictionary.
    If ``server_config`` is a path, the configuration is loaded and its
    ``base_dir`` key is set to the parent directory of the configuration file.

    Args:
        server_config: Optional path or directory. Defaults to "config.yml".

    Returns:
        A configuration dictionary.
    """
    server_config = server_config or "config.yml"
    if isinstance(server_config, str):
        config_path = server_config
        base_dir = os.path.dirname(config_path)
        if not base_dir:
            base_dir = get_res_test_dir()
            config_path = os.path.join(base_dir, config_path)
        elif not os.path.isabs(base_dir):
            base_dir = os.path.abspath(base_dir)
        with open(config_path, encoding="utf-8") as fp:
            server_config = yaml.safe_load(fp)
            assert isinstance(server_config, dict)
            server_config["base_dir"] = base_dir
    else:
        assert isinstance(server_config, collections.abc.Mapping)
    return server_config


def get_api_ctx(
    api_name: str,
    api_ctx_cls: type[T],
    server_config: Optional[Union[str, Mapping[str, Any]]] = None,
    framework: Optional[Framework] = None,
    extension_registry: Optional[ExtensionRegistry] = None,
) -> T:
    """Get the API context object for the given
    API name *api_name*, API context class *api_ctx_cls*,
    and server configuration path or dictionary *server_config*.

    This function is used for testing API contexts and controllers.

    Args:
        api_name: The name of the API, e.g. "auth"
        api_ctx_cls: The API context class
        server_config: Server configuration string or dictionary.
            If it is just a filename, it is resolved against test resource
            directory "${project}/test/webapi/res".
            If it is a relative path, it is resolved against the current
            working directory (not recommended).
            Defaults to ``'config.yml'``.
        framework: Web framework, defaults to a MockFramework.
        extension_registry: Extension registry,
            defaults to xcube's populated default extension registry

    Returns:
        The API context object

    Raises:
         AssertionError: if API context object can not be determined
    """
    server = get_server(
        server_config or "config.yml",
        framework=framework,
        extension_registry=extension_registry,
    )
    api_ctx = server.ctx.get_api_ctx(api_name, cls=api_ctx_cls)
    assert isinstance(api_ctx, api_ctx_cls)
    return api_ctx


def get_res_test_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "res"))


class RoutesTestCase(ServerTestCase):
    """Base class for xcube Server API tests."""

    # noinspection PyMethodMayBeStatic
    def get_config_filename(self) -> str:
        """Get configuration filename.
        Default impl. returns ``'config.yml'``."""
        return f"config.yml"

    # noinspection PyMethodMayBeStatic
    def get_config_path(self) -> str:
        """Get absolute path to configuration file.
        Default impl. uses ``self.get_config_filename()`` to construct
        a path into test resources.
        """
        return f"{get_res_test_dir()}/{self.get_config_filename()}"

    def get_config(self) -> Mapping[str, Any]:
        """Get configuration.
        Default impl. uses ``self.get_config_path()`` to load
        configuration from YAML file.
        Then sets 'base_dir' configuration parameter.
        """
        config_path = self.get_config_path()
        base_dir = os.path.dirname(config_path) or "."
        with open(config_path, encoding="utf-8") as fp:
            server_config = yaml.safe_load(fp)
            server_config["base_dir"] = base_dir
            return server_config

    def get_extension_registry(self):
        """Gets xcube default extension registry
        with all extensions loaded.
        """
        return get_extension_registry()
