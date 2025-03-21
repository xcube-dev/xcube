# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional


def split_var_assignment(var_name_or_assign: str) -> tuple[str, Optional[str]]:
    """Split *var_name_or_assign* into a variable name and expression part.

    Args:
        var_name_or_assign: A variable name or an expression

    Return:
        A pair (var_name, var_expr) if *var_name_or_assign* is an assignment
        expression, otherwise (var_name, None).
    """
    if "=" in var_name_or_assign:
        var_name, var_expr = map(
            lambda s: s.strip(), var_name_or_assign.split("=", maxsplit=1)
        )
        return var_name, var_expr
    else:
        return var_name_or_assign, None
