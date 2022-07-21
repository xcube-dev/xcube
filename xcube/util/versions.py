# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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

import sys
from typing import Dict
from typing import List

DEFAULT_DEPENDENCY_NAMES = \
    ['affine', 'click', 'cmocean', 'dask', 'dask - image', 'distributed',
     'fiona', 'fsspec', 'gdal', 'geopandas', 'jdcal', 'jsonschema',
     'matplotlib - base', 'netcdf4', 'numba', 'numpy', 'pandas', 'pillow',
     'pyjwt', 'pyproj', 'pyyaml', 'rasterio', 'requests',
     'requests-oauthlib', 's3fs', 'scipy', 'setuptools', 'shapely',
     'tornado', 'urllib3', 'xarray', 'zarr']

_XCUBE_VERSIONS = None


def get_xcube_versions() -> Dict[str, str]:
    """
    Get a mapping from xcube package names to package versions.

    :return: A mapping of the package names to package versions
    The result computed from the evaluating the expression.
    """
    from .plugin import get_plugins

    global _XCUBE_VERSIONS
    if _XCUBE_VERSIONS is None:
        plugin_names = [f'{plugin}.version'
                        for plugin in list(get_plugins().keys())]

        _XCUBE_VERSIONS = get_versions(DEFAULT_DEPENDENCY_NAMES,
                                       plugin_names)
    return _XCUBE_VERSIONS


def get_versions(dependency_names: List[str], plugin_names: List[str]) \
        -> Dict[str, str]:
    """
    Get a mapping from package names to package versions.
    The input is divided into names of packages that are external dependencies
    and into names of xcube plugins.

    :param dependency_names: A list of names of packages of which the versions
        shall be found
    :param plugin_names: A list of names of xcube plugins of which the versions
        shall be found
    :return: A mapping of the package names to package versions
    The result computed from the evaluating the expression.
    """
    # Idea borrowed from xarray.print_versions
    import importlib

    def _maybe_add_dot_version(name: str):
        if not name.endswith('.version'):
            return f'{name}.version'
        return name

    dependencies = [(_maybe_add_dot_version(plugin_name),
                     lambda mod: mod.version)
                    for plugin_name in plugin_names]

    def _find_module_version(mod):
        if hasattr(mod, '__version__'):
            return mod.__version__
        elif hasattr(mod, 'version'):
            return mod.version
        else:
            return 'unknown'

    dependencies += [(dependency_name, _find_module_version)
                     for dependency_name in dependency_names]

    dependencies_dict = {}
    for (module_name, module_version) in dependencies:
        module_key = module_name.split('.')[0]
        # noinspection PyBroadException
        try:
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                module = importlib.import_module(module_name)
        except BaseException:
            pass
        else:
            # noinspection PyBroadException
            try:
                dependencies_dict[module_key.split('.version')[0]] \
                    = module_version(module)
            except BaseException as e:
                dependencies_dict[module_key.split('.version')[0]] = "installed"

    return dependencies_dict
