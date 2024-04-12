# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

UNDEFINED_STR = "UNDEFINED"


class _Undefined:
    """Represents the UNDEFINED value."""

    _hash_code = hash(UNDEFINED_STR) + 1

    def __str__(self):
        return UNDEFINED_STR

    def __repr__(self):
        return UNDEFINED_STR

    def __eq__(self, other):
        return self is other or isinstance(other, _Undefined)

    def __hash__(self) -> int:
        return _Undefined._hash_code


#: Singleton value used to indicate an undefined state.
UNDEFINED = _Undefined()
