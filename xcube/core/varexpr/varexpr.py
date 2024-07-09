# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import ast
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Callable

from xcube.core.varexpr.error import VarExprError

Names = Mapping[str, Any]
UnaryFunction = Callable[[Any], Any]
BinaryFunction = Callable[[Any, Any], Any]
ComparisonFunction = Callable[[Any, Any], bool]

_UNARY_OPS: Mapping[str, UnaryFunction] = {
    "UAdd": lambda x: +x,
    "USub": lambda x: -x,
    "Invert": lambda x: ~x,
}
_BINARY_OPS: Mapping[str, BinaryFunction] = {
    "Add": lambda x, y: x + y,
    "Sub": lambda x, y: x - y,
    "Mult": lambda x, y: x * y,
    "Div": lambda x, y: x / y,
    "FloorDiv": lambda x, y: x // y,
    "Mod": lambda x, y: x % y,
    "Pow": lambda x, y: x**y,
    "LShift": lambda x, y: x << y,
    "RShift": lambda x, y: x >> y,
    "BitAnd": lambda x, y: x & y,
    "BitXor": lambda x, y: x ^ y,
    "BitOr": lambda x, y: x | y,
}

_COMPARISON_OPS: Mapping[str, ComparisonFunction] = {
    "Eq": lambda x, y: x == y,
    "NotEq": lambda x, y: x != y,
    "Lt": lambda x, y: x < y,
    "LtE": lambda x, y: x <= y,
    "Gt": lambda x, y: x > y,
    "GtE": lambda x, y: x >= y,
    "In": lambda x, y: x in y,
    "NotIn": lambda x, y: x not in y,
    "Is": lambda x, y: x is y,
    "IsNot": lambda x, y: x is not y,
}


def parse(code: str) -> "VarExpr":
    # noinspection PyTypeChecker
    expr_node: ast.expr = ast.parse(code, mode="eval")
    return VarExprFactory().visit(expr_node)


def evaluate(code: str, names: Mapping[str, Any]) -> Any:
    var_expr = parse(code)
    return var_expr.evaluate(names)


class VarExpr:
    """Represents a node of a variable expression."""

    @abstractmethod
    def evaluate(self, names: Names) -> Any:
        """Evaluate the expression."""


class Constant(VarExpr):
    # noinspection PyShadowingBuiltins
    def __init__(self, value: Any):
        self.value = value

    def evaluate(self, names: Names) -> Any:
        return self.value


class Name(VarExpr):
    # noinspection PyShadowingBuiltins
    def __init__(self, id: str):
        self.id = id

    def evaluate(self, names: Names) -> Any:
        try:
            return names[self.id]
        except KeyError:
            raise VarExprError(f"name {self.id!r} is not defined")


class Call(VarExpr):
    def __init__(
        self, func: VarExpr, args: list[VarExpr], keywords: Mapping[str, VarExpr]
    ):
        self.func = func
        self.args = args
        self.keywords = keywords

    def evaluate(self, names: Names) -> Any:
        return self.func.evaluate(names)(
            *(arg.evaluate(names) for arg in self.args),
            **{value: arg.evaluate(names) for value, arg in self.keywords.items()},
        )


class UnaryOp(VarExpr):
    def __init__(self, op: UnaryFunction, operand: VarExpr):
        self.op = op
        self.operand = operand

    def evaluate(self, names: Names) -> Any:
        return self.op(self.operand.evaluate(names))


class BinOp(VarExpr):
    def __init__(self, left: VarExpr, op: BinaryFunction, right: VarExpr):
        self.left = left
        self.op = op
        self.right = right

    def evaluate(self, names: Names) -> Any:
        return self.op(self.left.evaluate(names), self.right.evaluate(names))


class Compare(VarExpr):
    def __init__(
        self, left: VarExpr, ops: list[BinaryFunction], comparators: list[VarExpr]
    ):
        self.left = left
        self.ops = ops
        self.comparators = comparators

    def evaluate(self, names: Names) -> bool:
        left = self.left.evaluate(names)
        for op, comparator in zip(self.ops, self.comparators):
            right = comparator.evaluate(names)
            if not op(left, right):
                return False
            left = right
        return True


class BoolOp(VarExpr):
    def __init__(self, op: str, values: list[VarExpr]):
        self.op = op
        self.values = values

    def evaluate(self, names: Names) -> bool:
        if self.op == "Or":
            value = None
            for value in self.values:
                value = value.evaluate(names)
                if value:
                    break
            return value
        else:
            value = None
            for value in self.values:
                value = value.evaluate(names)
                if not value:
                    break
            return value


class IfExp(VarExpr):
    # noinspection SpellCheckingInspection
    def __init__(self, test: VarExpr, body: VarExpr, orelse: VarExpr):
        self.test = test
        self.body = body
        self.orelse = orelse

    def evaluate(self, names: Names) -> bool:
        return (
            self.body.evaluate(names)
            if self.test.evaluate(names)
            else self.orelse.evaluate(names)
        )


class VarExprFactory(ast.NodeVisitor):
    def visit(self, node: ast.expr) -> VarExpr:
        return super().visit(node)

    def generic_visit(self, node: ast.AST):
        raise VarExprError(
            f"unsupported expression node of type {node.__class__.__name__!r}"
        )

    def visit_Expression(self, node: ast.Expression):
        # noinspection PyTypeChecker
        return super().visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        return Constant(node.value)

    def visit_Name(self, node: ast.Name):
        return Name(node.id)

    def visit_Call(self, node: ast.Call):
        return Call(
            self.visit(node.func),
            [self.visit(arg) for arg in node.args],
            {keyword.arg: self.visit(keyword.value) for keyword in node.keywords},
        )

    def visit_UnaryOp(self, node: ast.UnaryOp):
        return UnaryOp(_UNARY_OPS[node.op.__class__.__name__], self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp):
        return BinOp(
            self.visit(node.left),
            _BINARY_OPS[node.op.__class__.__name__],
            self.visit(node.right),
        )

    def visit_Compare(self, node: ast.Compare):
        return Compare(
            self.visit(node.left),
            [_COMPARISON_OPS[op.__class__.__name__] for op in node.ops],
            [self.visit(c) for c in node.comparators],
        )

    def visit_BoolOp(self, node: ast.BoolOp):
        return BoolOp(node.op.__class__.__name__, [self.visit(v) for v in node.values])

    def visit_IfExp(self, node: ast.IfExp):
        return IfExp(
            self.visit(node.test), self.visit(node.body), self.visit(node.orelse)
        )
