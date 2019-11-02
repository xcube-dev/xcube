# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import abc
import importlib
import pkgutil
import time
import traceback
import warnings
from typing import Callable, Dict, Optional, Any

from pkg_resources import iter_entry_points

from xcube.util.extension import Extension

DEFAULT_PLUGIN_ENTRY_POINT_GROUP_NAME = 'xcube_plugins'
DEFAULT_PLUGIN_MODULE_PREFIX = 'xcube_'
DEFAULT_PLUGIN_MODULE_NAME = 'plugin'
DEFAULT_PLUGIN_MODULE_FUNCTION_NAME = 'init_plugin'
DEFAULT_PLUGIN_LOAD_TIME_LIMIT = 100  # milliseconds
DEFAULT_PLUGIN_INIT_TIME_LIMIT = 100  # milliseconds

#: Mapping of xcube entry point names to JSON-serializable plugin meta-information.
_PLUGIN_REGISTRY = None


def init_plugins() -> None:
    """Load plugins if not already done."""
    global _PLUGIN_REGISTRY
    if _PLUGIN_REGISTRY is None:
        _PLUGIN_REGISTRY = load_plugins()


def get_plugins() -> Dict[str, Dict]:
    """Get mapping of "xcube_plugins" entry point names to JSON-serializable plugin meta-information."""
    init_plugins()
    global _PLUGIN_REGISTRY
    return dict(_PLUGIN_REGISTRY)


def get_extension_registry():
    """Get populated extension registry."""
    from xcube.util.extension import get_extension_registry
    init_plugins()
    return get_extension_registry()


def discover_plugin_modules(module_prefixes=None):
    module_prefixes = module_prefixes or [DEFAULT_PLUGIN_MODULE_PREFIX]
    entry_points = []
    for module_finder, module_name, ispkg in pkgutil.iter_modules():
        if any([module_name.startswith(module_prefix) for module_prefix in module_prefixes]):
            # TODO (forman): Consider turning this into debug log:
            # print(f'xcube plugin module found: {module_name}')
            entry_points.append(_ModuleEntryPoint(module_name))
    return entry_points


def load_plugins(entry_points=None, ext_registry=None):
    if entry_points is None:
        entry_points = list(iter_entry_points(group=DEFAULT_PLUGIN_ENTRY_POINT_GROUP_NAME, name=None)) \
                       + discover_plugin_modules()

    if ext_registry is None:
        from xcube.util.extension import get_extension_registry
        ext_registry = get_extension_registry()

    plugins = {}

    for entry_point in entry_points:
        # TODO (forman): Consider turning this into debug log:
        # print(f'loading xcube plugin {entry_point.name!r}')

        t0 = time.perf_counter()

        # noinspection PyBroadException
        try:
            plugin_init_function = entry_point.load()
        except Exception as e:
            _handle_error(entry_point, e)
            continue

        millis = int(1000 * (time.perf_counter() - t0))
        if millis >= DEFAULT_PLUGIN_LOAD_TIME_LIMIT:
            warnings.warn(f'loading xcube plugin {entry_point.name!r} took {millis} ms, '
                          f'consider code optimization!')

        if not callable(plugin_init_function):
            # We use warning and not raise to allow loading xcube despite a broken plugin. Raise would stop xcube.
            warnings.warn(f'xcube plugin {entry_point.name!r} '
                          f'must be callable but got a {type(plugin_init_function)!r}')
            continue

        t0 = time.perf_counter()

        # noinspection PyBroadException
        try:
            plugin_init_function(ext_registry)
        except Exception as e:
            _handle_error(entry_point, e)
            continue

        millis = int(1000 * (time.perf_counter() - t0))
        if millis >= DEFAULT_PLUGIN_INIT_TIME_LIMIT:
            warnings.warn(f'initializing xcube plugin {entry_point.name!r} took {millis} ms, '
                          f'consider code optimization!')

        plugins[entry_point.name] = {'name': entry_point.name, 'doc': plugin_init_function.__doc__}

    return plugins


def _handle_error(entry_point, e):
    # We use warning and not raise to allow loading xcube despite a broken plugin. Raise would stop xcube.
    warnings.warn(f'Unexpected exception while loading xcube plugin {entry_point.name!r}: {e}')
    traceback.print_exc()


class _ModuleEntryPoint:
    def __init__(self, module_name: str):
        self._module_name = module_name

    @property
    def name(self) -> str:
        return self._module_name

    def load(self) -> Callable:
        """
        Load function "init_plugin()" either from "<module_name>.plugin" or "<module_name>"
        :return: plugin init function
        """
        module_name = self._module_name
        try:
            plugin_module = importlib.import_module(self._module_name + '.' + DEFAULT_PLUGIN_MODULE_NAME)
        except ModuleNotFoundError:
            plugin_module = importlib.import_module(self._module_name)

        module_func_name = DEFAULT_PLUGIN_MODULE_FUNCTION_NAME
        if hasattr(plugin_module, module_func_name):
            module_func = getattr(plugin_module, module_func_name)
            if callable(module_func):
                return module_func

        raise AttributeError(f'xcube plugin module {module_name!r} must define '
                             f'a function named {module_func_name!r}')


class ExtensionComponent(metaclass=abc.ABCMeta):
    """
    Utility base class for extension components.
    """

    def __init__(self, extension_point: str, name: str):
        if not extension_point:
            raise ValueError('extension_point must be given')
        if not name:
            raise ValueError('name must be given')
        self._extension_point = extension_point
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def extension_point(self) -> str:
        """
        :return: The extension point for this component.
        """
        return self._extension_point

    @property
    def extension(self) -> Optional[Extension]:
        """
        :return: The extension for this component. None, if it has not (yet) been registered as an extension.
        """
        return get_extension_registry().get_extension(self._extension_point, self._name)

    def get_metadata_attr(self, key: str, default: Any = None) -> Any:
        """
        :return: A metadata attribute for *key* and given *default* value.
        """
        extension = self.extension
        return extension.metadata.get(key, default) if extension is not None else ''
