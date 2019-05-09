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

from collections import OrderedDict
from typing import List, Type, Any


# noinspection PyShadowingBuiltins
class ObjRegistration:
    def __init__(self, registry: 'ObjRegistry', type: Type, name: str, obj: Any):
        if not isinstance(obj, type):
            raise ValueError(f'obj must be an instance of {type}')
        self._registry = registry
        self._type = type
        self._name = name
        self._obj = obj
        self._deleted = False

    @property
    def type(self) -> Type:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def obj(self) -> Any:
        return self._obj

    @property
    def deleted(self) -> bool:
        return self._deleted

    def delete(self):
        if not self._deleted:
            # noinspection PyProtectedMember
            self._registry._delete(self._name, type=self._type)
            self._deleted = True


# noinspection PyShadowingBuiltins
class ObjRegistry:

    def __init__(self):
        self._registrations = OrderedDict()

    def has(self, name: str, type: Type = object) -> bool:
        if type in self._registrations:
            return name in self._registrations[type]
        return False

    def get(self, name: str, type: Type = object) -> Any:
        return self._registrations[type][name].obj

    def get_all(self, type: Type = object) -> List[Any]:
        if type in self._registrations:
            return [v.obj for v in self._registrations[type].values()]
        return []

    def put(self, name: str, obj: Any, type: Type = object) -> ObjRegistration:
        if type in self._registrations:
            obj_dict = self._registrations[type]
        else:
            obj_dict = OrderedDict()
            self._registrations[type] = obj_dict
        registration = ObjRegistration(self, type, name, obj)
        obj_dict[name] = registration
        return registration

    def _delete(self, name: str, type: Type = object):
        obj_dict = self._registrations[type]
        del obj_dict[name]
        if not obj_dict:
            del self._registrations[type]


_OBJ_REGISTRY = ObjRegistry()


def get_obj_registry() -> ObjRegistry:
    return _OBJ_REGISTRY
