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
from typing import Any, Union, Tuple, Type, Container

_DEFAULT_NAME = 'value'


def assert_not_none(value: Any, name: str = None):
    if value is None:
        raise ValueError(f'{name or _DEFAULT_NAME} must not be None')


def assert_given(value: Any, name: str = None):
    if not value:
        raise ValueError(f'{name or _DEFAULT_NAME} must be given')


def assert_instance(value: Any, type: Union[Type, Tuple[Type, ...]], name: str = None):
    if not isinstance(value, type):
        raise ValueError(f'{name or _DEFAULT_NAME} must be an instance of {type}')


def assert_in(value: Any, container: Container, name: str = None):
    if value not in container:
        raise ValueError(f'{name or _DEFAULT_NAME} must be one of {container}')


def assert_condition(condition: Any, message: str):
    if not condition:
        raise ValueError(message)
