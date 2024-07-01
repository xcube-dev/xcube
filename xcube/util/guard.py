# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import Sequence
from typing import Optional

import inspect

_GUARD_ATTRS = {"_obj", "_attrs", "get"}


class Guard:
    def __init__(self, obj):
        self._obj = obj

    def get(self):
        return self._obj


def new_type_guard(type_: type, attrs: Optional[Sequence[str]] = None) -> type:

    class _Guard(Guard):
        _attrs = set(attrs) if attrs is not None else set()

        def __getattribute__(self, name: str):
            if name in _GUARD_ATTRS:
                return super().__getattribute__(name)
            if name not in _Guard._attrs:
                raise AttributeError(
                    f"attribute {name!r} of {self._obj.__class__.__name__!r}"
                    " object is protected"
                )
            return getattr(self._obj, name)

    _Guard.__name__ = f"{type_.__name__}Guard"

    for t in inspect.getmro(type_):
        print(f"Type {t!r}:")

        for k, v in t.__dict__.items():
            if k.startswith("__") and k.endswith("__"):
                # dunder
                print(f"  {k!r} --> {type(v)}")

    return _Guard
