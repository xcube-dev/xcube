# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.core.varexpr import VarExprContext, VarExprError

from ...server.api import ApiError
from .context import ExpressionsContext


# noinspection PyUnusedLocal
def get_expressions_capabilities(ctx: ExpressionsContext):
    return dict(
        namespace=dict(
            constants=VarExprContext.get_constants(),
            arrayFunctions=VarExprContext.get_array_functions(),
            otherFunctions=VarExprContext.get_other_functions(),
            arrayOperators=VarExprContext.get_array_operators(),
            otherOperators=VarExprContext.get_other_operators(),
        )
    )


def validate_expression(ctx: ExpressionsContext, ds_id: str, var_expr: str):
    dataset = ctx.datasets_ctx.get_dataset(ds_id)
    indexers = {
        str(dim_name): slice(0, min(2, dim_size))
        for dim_name, dim_size in dataset.sizes.items()
        if dim_size > 0
        and dim_name in dataset.coords
        and dataset.coords[dim_name].shape == (dim_size,)
    }
    var_expr_ctx = VarExprContext(dataset.isel(indexers))
    try:
        result = var_expr_ctx.evaluate(var_expr)
    except VarExprError as e:
        raise ApiError.BadRequest(f"{e}")
    try:
        return dict(result=result.mean().values.item())
    except BaseException as e:
        raise ApiError.BadRequest(f"{e}")
