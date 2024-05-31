# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Union, Tuple, Type
from collections.abc import Container

_DEFAULT_NAME = "value"


def assert_not_none(
    value: Any, name: str = None, exception_type: type[Exception] = ValueError
):
    """Assert *value* is not None.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        name: Name of a variable that holds *value*.
        exception_type: The exception type. Default is ``ValueError``.
    """
    if value is None:
        raise exception_type(f"{name or _DEFAULT_NAME} must not be None")


def assert_given(
    value: Any, name: str = None, exception_type: type[Exception] = ValueError
):
    """Assert *value* is not False when converted into a Boolean value.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        name: Name of a variable that holds *value*.
        exception_type: The exception type. Default is ``ValueError``.
    """
    if not value:
        raise exception_type(f"{name or _DEFAULT_NAME} must be given")


def assert_instance(
    value: Any,
    dtype: Union[type, tuple[type, ...]],
    name: str = None,
    exception_type: type[Exception] = TypeError,
):
    """Assert *value* is an instance of data type *dtype*.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        dtype: A type or tuple of types.
        name: Name of a variable that holds *value*.
        exception_type: The exception type. Default is ``TypeError``.
    """
    if not isinstance(value, dtype):
        raise exception_type(
            f"{name or _DEFAULT_NAME} "
            f"must be an instance of "
            f"{dtype}, was {type(value)}"
        )


def assert_subclass(
    value: Any,
    cls: Union[type, tuple[type, ...]],
    name: str = None,
    exception_type: type[Exception] = TypeError,
):
    """Assert *value* is a subclass of class *cls*.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        cls: A class or tuple of classes.
        name: Name of a variable that holds *value*.
        exception_type: The exception type. Default is ``TypeError``.
    """
    if not issubclass(value, cls):
        raise exception_type(
            f"{name or _DEFAULT_NAME} " f"must be a subclass of " f"{cls}, was {value}"
        )


def assert_in(
    value: Any,
    container: Container,
    name: str = None,
    exception_type: type[Exception] = ValueError,
):
    """Assert *value* is a member of *container*.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test for membership.
        container: The container.
        name: Name of a variable that holds *value*.
        exception_type: The exception type. Default is ``ValueError``.
    """
    if value not in container:
        raise exception_type(f"{name or _DEFAULT_NAME} " f"must be one of {container}")


def assert_true(value: Any, message: str, exception_type: type[Exception] = ValueError):
    """Assert *value* is true after conversion into a Boolean value.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        message: The error message used if the assertion fails.
        exception_type: The exception type. Default is ``ValueError``.
    """
    if not value:
        raise exception_type(message)


def assert_false(
    value: Any, message: str, exception_type: type[Exception] = ValueError
):
    """Assert *value* is false after conversion into a Boolean value.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        message: The error message used if the assertion fails.
        exception_type: The exception type. Default is ``ValueError``.
    """
    if value:
        raise exception_type(message)
