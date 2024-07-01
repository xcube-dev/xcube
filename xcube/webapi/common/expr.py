# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
import ast
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Callable, Optional

UnaryOpFn = Callable[[Any], Any]
BinOpFn = Callable[[Any, Any], Any]


class Expr:
    @abstractmethod
    def eval(self, ns: Mapping[str, Any]) -> Any:
        """Evaluate this expression."""

    @classmethod
    def parse(cls, code: str) -> "Expr":
        """Parse the given expression code."""
        node = ast.parse(code, mode="eval")
        compiler = ExprCompiler()
        return compiler.visit(node)


class Constant(Expr):
    def __init__(self, value: Any):
        self.value = value

    def eval(self, ns: Mapping[str, Any]) -> Any:
        return self.value


class List(Expr):
    def __init__(self, elts: list[Expr]):
        self.elts = elts

    def eval(self, ns: Mapping[str, Any]) -> list[Any]:
        return [elt.eval(ns) for elt in self.elts]


class Name(Expr):
    def __init__(self, id_: str):
        self.id = id_

    def eval(self, ns: Mapping[str, Any]) -> Any:
        if self.id not in ns:
            raise ExprEvalError(f"name {self.id!r} is not defined")
        return ns[self.id]


class UnaryOp(Expr):
    def __init__(self, op: UnaryOpFn, operand: Expr):
        self.op = op
        self.operand = operand

    def eval(self, ns: Mapping[str, Any]) -> Any:
        return self.op(self.operand.eval(ns))


class BinOp(Expr):
    def __init__(self, left: Expr, op: BinOpFn, right: Expr):
        self.left = left
        self.op = op
        self.right = right

    def eval(self, ns: Mapping[str, Any]) -> Any:
        return self.op(self.left.eval(ns), self.right.eval(ns))


class Compare(Expr):
    def __init__(self, left: Expr, ops: list[BinOpFn], comparators: list[Expr]):
        self.left = left
        self.ops = ops
        self.comparators = comparators

    def eval(self, ns: Mapping[str, Any]) -> Any:
        left = self.left.eval(ns)
        for op, comparator in zip(self.ops, self.comparators):
            right = comparator.eval(ns)
            if not op(left, right):
                return False
            left = right
        return True


class IfExp(Expr):
    def __init__(self, test: Expr, body: Expr, orelse: Expr):
        self.test = test
        self.body = body
        self.orelse = orelse

    def eval(self, ns: Mapping[str, Any]) -> Any:
        return self.body.eval(ns) if self.test.eval(ns) else self.orelse.eval(ns)


class Call(Expr):
    def __init__(self, func: Expr, args: list[Expr], keywords: dict[str, Expr]):
        self.func = func
        self.args = args
        self.keywords = keywords

    def eval(self, ns: Mapping[str, Any]) -> Any:
        func = self.func.eval(ns)
        args = [arg.eval(ns) for arg in self.args]
        keywords = {arg: value.eval(ns) for arg, value in self.keywords.items()}
        return func(*args, **keywords)


class Attribute(Expr):
    def __init__(self, value: Expr, attr: str):
        self.value = value
        self.attr = attr

    def eval(self, ns: Mapping[str, Any]) -> Any:
        value = self.value.eval(ns)
        return getattr(value, self.attr)


class Subscript(Expr):
    def __init__(self, value: Expr, slice: Expr):
        self.value = value
        self.slice = slice

    def eval(self, ns: Mapping[str, Any]) -> Any:
        value = self.value.eval(ns)
        slice = self.slice.eval(ns)
        return value[slice]


class Slice(Expr):
    def __init__(
        self, lower: Optional[Expr], upper: Optional[Expr], step: Optional[Expr]
    ):
        self.lower = lower
        self.upper = upper
        self.step = step

    def eval(self, ns: Mapping[str, Any]) -> Any:
        lower = None if self.lower is None else self.lower.eval(ns)
        upper = None if self.upper is None else self.upper.eval(ns)
        step = None if self.step is None else self.step.eval(ns)
        return slice(lower, upper, step)


class And(Expr):
    def __init__(self, values: list[Expr]):
        self.values = values

    def eval(self, ns: Mapping[str, Any]) -> Any:
        result = None
        for value in self.values:
            result = value.eval(ns)
            if not result:
                break
        return result


class Or(Expr):
    def __init__(self, values: list[Expr]):
        self.values = values

    def eval(self, ns: Mapping[str, Any]) -> Any:
        result = None
        for value in self.values:
            result = value.eval(ns)
            if result:
                break
        return result


class ExprEvalError(ValueError):
    pass


class ExprCompiler(ast.NodeVisitor):
    unary_ops: dict[str, UnaryOpFn] = {
        "UAdd": lambda a: +a,
        "USub": lambda a: -a,
        "Invert": lambda a: ~a,
        "Not": lambda a: not a,
    }

    bin_ops: dict[str, BinOpFn] = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mult": lambda a, b: a * b,
        "MatMult": lambda a, b: a @ b,
        "Div": lambda a, b: a / b,
        "Mod": lambda a, b: a % b,
        "Pow": lambda a, b: a**b,
        "LShift": lambda a, b: a << b,
        "RShift": lambda a, b: a >> b,
        "BitOr": lambda a, b: a | b,
        "BitXor": lambda a, b: a ^ b,
        "BitAnd": lambda a, b: a & b,
        "FloorDiv": lambda a, b: a // b,
    }
    cmp_ops: dict[str, BinOpFn] = {
        "Eq": lambda a, b: a == b,
        "NotEq": lambda a, b: a != b,
        "Lt": lambda a, b: a < b,
        "LtE": lambda a, b: a <= b,
        "Gt": lambda a, b: a > b,
        "GtE": lambda a, b: a >= b,
        "Is": lambda a, b: a is b,
        "IsNot": lambda a, b: a is not b,
        "In": lambda a, b: a in b,
        "NotIn": lambda a, b: a not in b,
    }

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        return Constant(node.value)

    def visit_List(self, node: ast.List):
        return List([self.visit(elt) for elt in node.elts])

    def visit_Name(self, node: ast.Name):
        return Name(node.id)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        unary_op = self.unary_ops[node.op.__class__.__name__]
        operand = self.visit(node.operand)
        return UnaryOp(unary_op, operand)

    def visit_BinOp(self, node: ast.BinOp):
        bin_op = self.bin_ops[node.op.__class__.__name__]
        left = self.visit(node.left)
        right = self.visit(node.right)
        return BinOp(left, bin_op, right)

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        ops = [self.cmp_ops[op.__class__.__name__] for op in node.ops]
        comparators = [self.visit(comparator) for comparator in node.comparators]
        return Compare(left, ops, comparators)

    def visit_BoolOp(self, node: ast.BoolOp):
        values = [self.visit(value) for value in node.values]
        return And(values) if isinstance(node.op, ast.And) else Or(values)

    def visit_IfExp(self, node: ast.IfExp):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return IfExp(test, body, orelse)

    def visit_Call(self, node: ast.Call):
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        keywords = {keyword.arg: self.visit(keyword.value) for keyword in node.keywords}
        return Call(func, args, keywords)

    def visit_Attribute(self, node: ast.Attribute):
        value = self.visit(node.value)
        return Attribute(value, node.attr)

    def visit_Subscript(self, node: ast.Subscript):
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return Subscript(value, slice)

    def visit_Slice(self, node: ast.Slice):
        lower = None if node.lower is None else self.visit(node.lower)
        upper = None if node.upper is None else self.visit(node.upper)
        step = None if node.step is None else self.visit(node.step)
        return Slice(lower, upper, step)

    def generic_visit(self, node: ast.AST):
        if isinstance(node, ast.AST):
            raise SyntaxError(f"unsupported expression ({node.__class__.__name__})")
        else:
            raise RuntimeError(f"received node of unexpected type {type(node)}")


def get_safe_numpy_funcs() -> dict[str, Callable]:
    import numpy

    return {
        k: v
        for k, v in numpy.__dict__.items()
        if isinstance(v, numpy.ufunc) and isinstance(k, str) and not k.startswith("_")
    }


def get_safe_python_funcs() -> dict[str, Callable]:
    import builtins

    return {
        k: getattr(builtins, k)
        for k in [
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "complex",
            "dict",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "hash",
            "hex",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "oct",
            "ord",
            "pow",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "slice",
            "sorted",
            "str",
            "sum",
            "tuple",
            "type",
            "zip",
        ]
    }
