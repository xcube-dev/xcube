# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import inspect
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

from xcube.util.assertions import assert_true

GET_EXTENSIONS_ATTR_NAME = "get_extensions"

Base = TypeVar("Base")
Ext = TypeVar("Ext")

BaseClass = type[Base]
ExtClass = type[Ext]

ExtClassItem = tuple[str, ExtClass]
ExtInstItem = tuple[str, Ext]


def extend(
    base_class: BaseClass,
    name: str,
    doc: Optional[str] = None,
    class_handler: Union[None, str, Callable[[ExtClassItem], Any]] = None,
    inst_handler: Union[None, str, Callable[[ExtInstItem], Any]] = None,
    ext_args: Optional[Sequence[Any]] = None,
    ext_kwargs: Optional[Mapping[str, Any]] = None,
) -> Callable[[ExtClass], ExtClass]:
    """A class decorator factory that adds an instance property named
    *property_name* to an existing class *base_class*. The value of the
    new property is an instance of the decorated class.

    This decorator can be used to add new functionality to existing
    classes without deriving from the existing class.

    The optional *class_handler* callback function can be used for
    registering the extensions in the base class and/or for
    validating the passed extension name and class.
    Likewise, the optional *inst_handler* callback function can be used for
    registering the extensions in the base class instances and/or for
    validating the passed extension name and instances.

    The decorated class is expected to have a constructor whose first
    positional argument is an instance of *base_class*.
    Other positional and keyword arguments may follow
    and may be passed to this decorator using the *ext_args*
    and *ext_kwargs* keyword arguments.

    Args:
        base_class: The base class to be extended.
        name: The new property's name.
        doc: The new property's docstring, optional.
            If omitted, the docstring of the decorated class, if any, is used.
        class_handler: Optional class handler.
            If given, it must be a callable that is called
            with a tuple comprising extension name and class.
            If this is a string, it is the name of a static or class method
            of the *base_class* which is called with that tuple.
        inst_handler: Optional instance handler.
            If given, it must be a callable that is called
            with a tuple comprising extension name and instance.
            If this is a string, it is the name of an instance method
            of the *base_class* which is called with that tuple.
        ext_args: positional arguments passed
            to the decorated class' constructor
        ext_kwargs: keyword arguments passed
            to the decorated class' constructor

    Returns: a class decorator function
    """
    assert_true(
        inspect.isclass(base_class),
        message=f"base_class must be a class type, but was {type(base_class).__name__}",
    )
    assert_true(
        isinstance(name, str) and name.isidentifier(),
        message=f"name must be a valid identifier, but was {name!r}",
    )
    assert_true(doc is None or isinstance(doc, str), message="doc must be a string")

    _name = "_" + name
    lock = threading.RLock()

    def call_handler(base, handler, ext):
        if isinstance(handler, str):
            getattr(base, handler)((name, ext))
        elif callable(handler):
            handler((name, ext))

    def add_class_property(ext_class: ExtClass) -> ExtClass:
        assert_true(
            inspect.isclass(ext_class),
            message="the extend() decorator can be used with classes only",
        )

        def _get_property(base: Base):
            ext = getattr(base, _name, None)
            if ext is None:
                with lock:
                    ext = getattr(base, _name, None)
                    if ext is None:
                        ext = ext_class(base, *(ext_args or ()), **(ext_kwargs or {}))
                        setattr(base, _name, ext)
                        call_handler(base, inst_handler, ext)
            return ext

        assert_true(
            not hasattr(base_class, name),
            message=f"a property named {name} already"
            f" exists in class {base_class.__name__}",
        )

        setattr(
            base_class,
            name,
            property(
                _get_property, None, None, ext_class.__doc__ if doc is None else doc
            ),
        )

        call_handler(base_class, class_handler, ext_class)

        return ext_class

    return add_class_property
