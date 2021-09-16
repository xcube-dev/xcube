# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import os
import sys
import yaml
from typing import Dict
from typing import List
from .plugin import get_plugins

DEPENDENCY_NAMES = \
    ['affine', 'click', 'cmocean', 'dask', 'dask - image', 'distributed',
     'fiona', 'fsspec', 'gdal', 'geopandas', 'jdcal', 'jsonschema',
     'matplotlib - base', 'netcdf4', 'numba', 'numpy', 'pandas', 'pillow',
     'pyjwt', 'pyproj', 'pyyaml', 'rasterio', 'requests',
     'requests - oauthlib', 's3fs', 'scipy', 'setuptools', 'shapely',
     'tornado', 'urllib3', 'xarray', 'zarr']


def get_xcube_dependencies() -> Dict[str, str]:
    """
    Get a mapping from package names to package versions.
    Lists all dependencies stated in the environment and
    all plugin packages.
    """
    # Idea stolen from xarray.print_versions

    # try to parse dependencies from environment.yml. If it can't be found,
    # use hard-coded list of dependencies
    environment_file = f'{os.path.dirname(os.path.abspath(__file__))}' \
                       f'../../../environment.yml'
    environment_file = os.path.abspath(environment_file)
    if os.path.exists(environment_file):
        with open(environment_file, 'r') as environment:
            env_dict = yaml.safe_load(environment)
            dependency_names = [dependency.split('=')[0].split('>')[0].strip()
                                for dependency in
                                env_dict.get('dependencies', [])]
    else:
        dependency_names = DEPENDENCY_NAMES

    plugin_names = [f'{plugin}.version'
                    for plugin in list(get_plugins().keys())]

    return get_dependencies(dependency_names, plugin_names)


def get_dependencies(dependency_names: List[str], plugin_names: List[str]) \
        -> Dict[str, str]:
    """
    Get a mapping from package names to package versions.
    Lists all dependencies stated in the environment and
    all plugin packages.
    """
    # Idea stolen from xarray.print_versions
    import importlib

    def _maybe_add_dot_version(name: str):
        if not name.endswith('.version'):
            return f'{name}.version'
        return name

    dependencies = [(_maybe_add_dot_version(plugin_name),
                     lambda mod: mod.version)
                    for plugin_name in plugin_names]
    dependencies += [(dependency_name, lambda mod: mod.__version__)
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
        except Exception:
            pass
        else:
            # noinspection PyBroadException
            try:
                dependencies_dict[module_key.split('.version')[0]] \
                    = module_version(module)
            except Exception as e:
                print(e)
                dependencies_dict[module_key.split('.version')[0]] = "installed"

    return dependencies_dict
