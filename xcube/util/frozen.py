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
from typing import (Generic, TypeVar, Tuple, Any,
                    Mapping, Optional, Iterable, List)

K = TypeVar('K')
V = TypeVar('V')


class FrozenDict(Generic[K, V], dict[K, V]):

    @classmethod
    def deep(cls, other: dict) -> "FrozenDict":
        return FrozenDict({key: freeze_value(value)
                           for key, value in other.items()})

    def clear(self) -> None:
        raise self.__forbidden()

    def popitem(self) -> Tuple[K, V]:
        raise self.__forbidden()

    def update(self, mapping: Mapping[K, V], **kwargs: V) -> None:
        raise self.__forbidden()

    def __setitem__(self, key: K, value: V) -> None:
        raise self.__forbidden(f'key {key!r}')

    def __delitem__(self, key: K) -> None:
        raise self.__forbidden(f'key {key!r}')

    def pop(self, key: K) -> V:
        raise self.__forbidden(f'key {key!r}')

    def __setattr__(self, name: str, value: Any) -> None:
        raise self.__forbidden(f'attribute {name!r}')

    def __delattr__(self, name: str) -> None:
        raise self.__forbidden(f'attribute {name!r}')

    def __forbidden(self, msg: Optional[str] = None):
        return TypeError('dict is read-only' + (': ' + msg if msg else ''))


class FrozenList(Generic[V], list[V]):

    @classmethod
    def deep(cls, other: List[V]) -> "FrozenList":
        return FrozenList([freeze_value(item) for item in other])

    def append(self, element: V):
        raise self.__forbidden()

    def clear(self):
        raise self.__forbidden()

    def extend(self, elements: Iterable[V]):
        raise self.__forbidden()

    def insert(self, index: int, elements: Iterable[V]):
        raise self.__forbidden()

    def pop(self, *args, **kwargs):
        raise self.__forbidden()

    def remove(self, value: V):
        raise self.__forbidden()

    def reverse(self):
        raise self.__forbidden()

    def sort(self, *args, **kwargs):
        raise self.__forbidden()

    def __iadd__(self, arg: Any):
        raise self.__forbidden()

    def __imul__(self, arg: Any):
        raise self.__forbidden()

    def __setitem__(self, index: int, value: V):
        raise self.__forbidden(f'index {index}')

    def __forbidden(self, msg: Optional[str] = None):
        return TypeError('list is read-only' + (': ' + msg if msg else ''))


def freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict.deep(value)
    if isinstance(value, collections.abc.Mapping):
        return FrozenDict.deep(dict(value))
    if isinstance(value, list):
        return FrozenList.deep(value)
    if isinstance(value, (tuple, set)):
        return FrozenList(list(value))
    return value
