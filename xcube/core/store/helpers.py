# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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
from typing import Optional, Mapping, Any, Type, Tuple

from xcube.constants import EXTENSION_POINT_CUBE_STORES
from xcube.core.store.store import CubeStore
from xcube.util.extension import ExtensionRegistry
from xcube.util.plugin import get_extension_registry


def new_cube_store(cube_store_id: str,
                   cube_store_params: Mapping[str, Any],
                   cube_store_requirements: Tuple[Type, ...] = None,
                   extension_registry: Optional[ExtensionRegistry] = None):
    extension_registry = extension_registry or get_extension_registry()
    if not extension_registry.has_extension(EXTENSION_POINT_CUBE_STORES, cube_store_id):
        raise ValueError(f'Unknown cube store "{cube_store_id}"')
    cube_store_class = extension_registry.get_component(EXTENSION_POINT_CUBE_STORES, cube_store_id)
    if not issubclass(cube_store_class, CubeStore):
        raise ValueError(f'Identifier "{cube_store_id}" does not register a valid cube store')
    if cube_store_requirements and not issubclass(cube_store_class, cube_store_requirements):
        raise ValueError(f'Cube store "{cube_store_id}" does not meet requirements')
    cube_store_params_schema = cube_store_class.get_cube_store_params_schema()
    cube_store_params = cube_store_params_schema.from_instance(cube_store_params) \
        if cube_store_params else {}
    return cube_store_class(**cube_store_params)
