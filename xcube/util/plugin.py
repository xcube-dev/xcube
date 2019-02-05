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

import traceback
import warnings

from pkg_resources import iter_entry_points


def load_plugins(entry_points, plugins=None):
    plugins = plugins if plugins is not None else {}
    for entry_point in entry_points:
        # noinspection PyBroadException
        try:
            plugin_init_function = entry_point.load()
        except Exception as e:
            _handle_error(entry_point, e)
            continue

        if callable(plugin_init_function):
            # noinspection PyBroadException
            try:
                plugin_init_function()
            except Exception as e:
                _handle_error(entry_point, e)
                continue
        else:
            warnings.warn(f'xcube plugin with entry point {entry_point.name!r} '
                          f'must be a callable but got a {type(plugin_init_function)!r}')
            continue

        # Here: use pkg_resources and introspection to generate a
        # JSON-serializable dictionary of plugin meta-information
        plugins[entry_point.name] = {'entry_point': entry_point.name}

    return plugins


def get_plugins():
    """Get mapping of "xcube_plugins" entry point names to JSON-serializable plugin meta-information."""
    return dict(_PLUGIN_REGISTRY)


def _handle_error(entry_point, e):
    warnings.warn('Unexpected exception while loading xcube plugin '
                  f'with entry point {entry_point.name!r}: {e}')
    traceback.print_exc()


#: Mapping of Cate entry point names to JSON-serializable plugin meta-information.
_PLUGIN_REGISTRY = load_plugins(iter_entry_points(group='xcube_plugins', name=None))
