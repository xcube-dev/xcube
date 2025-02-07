# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional, Tuple, Type, TypeVar, Union

from xcube.util.assertions import assert_true

T = TypeVar("T")

ItemType = Union[type[T], tuple[type[T], ...]]
Pair = tuple[T, T]
ScalarOrPair = Union[T, Pair]


def normalize_scalar_or_pair(
    value: ScalarOrPair[T],
    *,
    item_type: Optional[ItemType[T]] = None,
    name: Optional[str] = None,
) -> Pair:
    try:
        assert_true(
            len(value) <= 2,
            message=f"{name or 'Value'} must be a scalar or pair of "
            f"{item_type or 'scalars'}, was '{value}'",
        )
        x, y = value
    except TypeError:
        x, y = value, value
    if item_type is not None:
        assert_true(
            isinstance(x, item_type) and isinstance(y, item_type),
            message=f"{name or 'Value'} must be a scalar or pair of "
            f"{item_type}, was '{value}'",
        )
    return x, y
