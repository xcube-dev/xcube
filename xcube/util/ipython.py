# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import warnings
from typing import Type


def register_json_formatter(cls: type, to_dict_method_name: str = "to_dict"):
    """TODO

    Args:
        cls
        to_dict_method_name

    Returns:

    """
    if not hasattr(cls, to_dict_method_name) or not callable(
        getattr(cls, to_dict_method_name)
    ):
        raise ValueError(f"{cls} must define a {to_dict_method_name}() method")

    try:
        import IPython
        import IPython.display

        if IPython.get_ipython() is not None:

            def obj_to_dict(obj):
                return getattr(obj, to_dict_method_name)()

            ipy_formatter = IPython.get_ipython().display_formatter.formatters[
                "application/json"
            ]
            ipy_formatter.for_type(cls, obj_to_dict)

    except ImportError:
        pass


def enable_asyncio():
    """Enable asyncio package to be executable in Jupyter Notebooks."""
    try:
        import IPython

        if IPython.get_ipython() is not None:
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                warnings.warn(
                    "nest-asyncio required to use asyncio in Jupyter Notebooks"
                )
    except ImportError:
        pass
