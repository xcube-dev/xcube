# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import inspect
from typing import Callable, Any, Dict

from xcube.server.api import ApiError

from .context import ComputeContext
from .op.info import OpInfo


def get_compute_operations(ctx: ComputeContext):
    ops = ctx.op_registry.ops
    return {"operations": [encode_op(op_id, f) for op_id, f in ops.items()]}


def get_compute_operation(ctx: ComputeContext, op_id: str):
    op = ctx.op_registry.get_op(op_id)
    if op is None:
        raise ApiError.NotFound(f"operation {op_id!r} not found")
    return encode_op(op_id, op)


def encode_op(op_id: str, op: Callable) -> dict[str, Any]:
    op_info = OpInfo.get_op_info(op)
    op_json = {"operationId": op_id, "parametersSchema": op_info.params_schema}
    doc = inspect.getdoc(op)
    if doc:
        op_json.update(description=doc)
    return op_json
