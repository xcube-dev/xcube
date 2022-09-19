# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

import collections.abc
from typing import Generic, TypeVar, Tuple, Any, Iterable

K = TypeVar('K')
V = TypeVar('V')

_DICT_IS_READONLY = 'dict is read-only'
_LIST_IS_READONLY = 'list is read-only'


class FrozenDict(Generic[K, V], dict[K, V]):

    @classmethod
    def deep(cls, other: collections.abc.Mapping) -> "FrozenDict":
        return FrozenDict({key: freeze_value(value)
                           for key, value in other.items()})

    def clear(self) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def pop(self, *args, **kwargs) -> V:
        raise TypeError(_DICT_IS_READONLY)

    def popitem(self) -> Tuple[K, V]:
        raise TypeError(_DICT_IS_READONLY)

    def update(self, *args, **kwargs) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def __setitem__(self, key: K, value: V) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def __delitem__(self, key: K) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(_DICT_IS_READONLY)

    def __delattr__(self, name: str) -> None:
        raise TypeError(_DICT_IS_READONLY)


class FrozenList(Generic[V], list[V]):

    @classmethod
    def deep(cls, other: collections.abc.Sequence[V]) -> "FrozenList":
        return FrozenList([freeze_value(item) for item in other])

    def append(self, element: V):
        raise TypeError(_LIST_IS_READONLY)

    def extend(self, elements: Iterable[V]):
        raise TypeError(_LIST_IS_READONLY)

    def clear(self):
        raise TypeError(_LIST_IS_READONLY)

    def insert(self, index: int, elements: Iterable[V]):
        raise TypeError(_LIST_IS_READONLY)

    def pop(self, *args, **kwargs):
        raise TypeError(_LIST_IS_READONLY)

    def remove(self, value: V):
        raise TypeError(_LIST_IS_READONLY)

    def reverse(self):
        raise TypeError(_LIST_IS_READONLY)

    def sort(self, *args, **kwargs):
        raise TypeError(_LIST_IS_READONLY)

    def __iadd__(self, arg: Any):
        raise TypeError(_LIST_IS_READONLY)

    def __imul__(self, arg: Any):
        raise TypeError(_LIST_IS_READONLY)

    def __setitem__(self, index: int, value: V):
        raise TypeError(_LIST_IS_READONLY)


_PRIMITIVES = type(None), bool, int, float, complex, str


def freeze_value(value: Any) -> Any:
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, collections.abc.Mapping):
        return FrozenDict.deep(value)
    if isinstance(value, collections.abc.Sequence):
        return FrozenList.deep(value)
    return value
