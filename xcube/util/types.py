# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

from typing import Optional, Tuple, Type, TypeVar, Union

from xcube.util.assertions import assert_true

T = TypeVar('T', int, float)

ItemType = Union[Type[T], Tuple[Type[T], ...]]
Pair = Tuple[T, T]
ScalarOrPair = Union[T, Pair]


def normalize_scalar_or_pair(
        value: ScalarOrPair[T],
        *,
        item_type: Optional[ItemType[T]] = None,
        default: Optional[ScalarOrPair] = None,
        name: Optional[str] = None
) -> Pair:
    if value is None and default is not None:
        # not returning default immediately so that default is checked for
        # validity
        value = default
    try:
        assert_true(len(value) <= 2,
                    message=f"{name or 'Value'} must be a scalar or pair of "
                            f"{item_type or 'scalars'}, was '{value}'")
        x, y = value
    except TypeError:
        x, y = value, value
    if item_type is not None:
        x = _maybe_convert(x, item_type)
        y = _maybe_convert(y, item_type)
        assert_true(isinstance(x, item_type) and isinstance(y, item_type),
                    message=f"{name or 'Value'} must be a scalar or pair of "
                            f"{item_type}, was '{value}'")
    return x, y


def _maybe_convert(value: T, item_type: ItemType[T]) -> T:
    if not isinstance(value, item_type):
        try:
            for t in item_type:
                try:
                    value = t(value)
                    break
                except TypeError:
                    continue
        except TypeError:
            try:
                value = item_type(value)
            except TypeError:
                # leave value as is
                value = value
    return value
