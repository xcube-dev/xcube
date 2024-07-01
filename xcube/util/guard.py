# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

PRIMITIVE_TYPES = (type(None), bool, int, float, complex, str)


class Guard:
    def __init__(
        self, obj: object, attrs: set[str], types: list[tuple[type, set[str]]]
    ):
        self._obj = obj
        self._is_primitive = isinstance(obj, PRIMITIVE_TYPES)
        self._type_name = obj.__class__.__name__
        self._attrs = attrs
        self._types = types

    def __getattr__(self, attr_name: str):
        if not self._is_primitive and (attr_name not in self._attrs):
            return AttributeError(f"")
        value = getattr(self._obj, attr_name)
        return self._verify_value(value)

    def __call__(self, *args, **kwargs):
        # noinspection PyCallingNonCallable
        value = self._obj(*args, **kwargs)
        return self._verify_value(value)

    def __next__(self):
        # noinspection PyTypeChecker
        value = next(self._obj)
        return self._verify_value(value)

    def _verify_value(self, value):
        if isinstance(value, Guard):
            return value

        if isinstance(value, PRIMITIVE_TYPES):
            return value

        for t, attrs in self._types:
            if isinstance(value, t):
                return Guard(value, attrs, self._types)

        raise ValueError(
            f"encountered illegal value of type {value.__class__.__name__!r}"
        )
