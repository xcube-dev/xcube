# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import pytest

from xcube.webapi.common.expr import Attribute
from xcube.webapi.common.expr import BinOp
from xcube.webapi.common.expr import Call
from xcube.webapi.common.expr import Compare
from xcube.webapi.common.expr import Constant
from xcube.webapi.common.expr import Expr
from xcube.webapi.common.expr import IfExp
from xcube.webapi.common.expr import List
from xcube.webapi.common.expr import Name
from xcube.webapi.common.expr import Subscript
from xcube.webapi.common.expr import UnaryOp
from xcube.webapi.common.expr import get_safe_numpy_funcs
from xcube.webapi.common.expr import get_safe_python_funcs


class ExprTest(unittest.TestCase):
    def test_constant(self):
        ns = dict()

        expr = Expr.parse("...")
        self.assertIsInstance(expr, Constant)
        self.assertIs(Ellipsis, expr.eval(ns))

        expr = Expr.parse("None")
        self.assertIsInstance(expr, Constant)
        self.assertIs(None, expr.eval(ns))

        expr = Expr.parse("True")
        self.assertIsInstance(expr, Constant)
        self.assertIs(True, expr.eval(ns))

        expr = Expr.parse("False")
        self.assertIsInstance(expr, Constant)
        self.assertIs(False, expr.eval(ns))

        expr = Expr.parse("23")
        self.assertIsInstance(expr, Constant)
        self.assertEqual(23, expr.eval(ns))

        expr = Expr.parse("0.05")
        self.assertIsInstance(expr, Constant)
        self.assertEqual(0.05, expr.eval(ns))

        expr = Expr.parse("'hello!'")
        self.assertIsInstance(expr, Constant)
        self.assertEqual("hello!", expr.eval(ns))

    def test_name(self):
        ns = dict(a=1, b=2)

        expr = Expr.parse("a")
        self.assertIsInstance(expr, Name)
        self.assertEqual(1, expr.eval(ns))

        expr = Expr.parse("b")
        self.assertIsInstance(expr, Name)
        self.assertEqual(2, expr.eval(ns))

        expr = Expr.parse("c")
        self.assertIsInstance(expr, Name)
        with pytest.raises(ValueError, match="name 'c' is not defined"):
            expr.eval(ns)

    def test_call(self):

        def my_func(x):
            return 2 * x

        ns = dict(my_func=my_func)
        expr = Expr.parse("my_func(4)")
        self.assertIsInstance(expr, Call)
        self.assertEqual(8, expr.eval(ns))

        def my_func(x, y=0):
            return 2 * x + y

        ns = dict(my_func=my_func)
        expr = Expr.parse("my_func(4, y=2)")
        self.assertIsInstance(expr, Call)
        self.assertEqual(10, expr.eval(ns))

    def test_attribute(self):

        class Obj:
            x = 5

        ns = dict(my_obj=Obj())
        expr = Expr.parse("my_obj.x")
        self.assertIsInstance(expr, Attribute)
        self.assertEqual(5, expr.eval(ns))

    def test_subscript(self):
        ns = dict(my_list=[1, 2, 3, 4])

        expr = Expr.parse("my_list[2]")
        self.assertIsInstance(expr, Subscript)
        self.assertEqual(3, expr.eval(ns))

        expr = Expr.parse("my_list[-1]")
        self.assertIsInstance(expr, Subscript)
        self.assertEqual(4, expr.eval(ns))

    def test_slice(self):
        ns = dict(my_list=(1, 2, 3, 4, 5, 6))

        expr = Expr.parse("my_list[:]")
        self.assertIsInstance(expr, Subscript)
        self.assertEqual((1, 2, 3, 4, 5, 6), expr.eval(ns))

        expr = Expr.parse("my_list[1:4]")
        self.assertIsInstance(expr, Subscript)
        self.assertEqual((2, 3, 4), expr.eval(ns))

        expr = Expr.parse("my_list[2:]")
        self.assertIsInstance(expr, Subscript)
        self.assertEqual((3, 4, 5, 6), expr.eval(ns))

        expr = Expr.parse("my_list[:3]")
        self.assertIsInstance(expr, Subscript)
        self.assertEqual((1, 2, 3), expr.eval(ns))

        expr = Expr.parse("my_list[::2]")
        self.assertIsInstance(expr, Subscript)
        self.assertEqual((1, 3, 5), expr.eval(ns))

        expr = Expr.parse("my_list[::-1]")
        self.assertIsInstance(expr, Subscript)
        self.assertEqual((6, 5, 4, 3, 2, 1), expr.eval(ns))

    def test_list(self):
        ns = dict(a=1, b=2, c=3)
        expr = Expr.parse("[a, b, c, 4, 5]")
        self.assertIsInstance(expr, List)
        self.assertEqual([1, 2, 3, 4, 5], expr.eval(ns))

    def test_unary_op(self):
        ns = dict(a=1, b=2)

        expr = Expr.parse("+b")
        self.assertIsInstance(expr, UnaryOp)
        self.assertEqual(2, expr.eval(ns))

        expr = Expr.parse("-a")
        self.assertIsInstance(expr, UnaryOp)
        self.assertEqual(-1, expr.eval(ns))

        expr = Expr.parse("~a")
        self.assertIsInstance(expr, UnaryOp)
        self.assertEqual(-2, expr.eval(ns))

        expr = Expr.parse("not a")
        self.assertIsInstance(expr, UnaryOp)
        self.assertEqual(0, expr.eval(ns))

    def test_bin_op(self):
        ns = dict(a=1, b=2, c=3)

        expr = Expr.parse("a + b")
        self.assertIsInstance(expr, BinOp)
        self.assertEqual(3, expr.eval(ns))

        expr = Expr.parse("a - b")
        self.assertIsInstance(expr, BinOp)
        self.assertEqual(-1, expr.eval(ns))

        expr = Expr.parse("a / b")
        self.assertIsInstance(expr, BinOp)
        self.assertEqual(0.5, expr.eval(ns))

        expr = Expr.parse("c // b")
        self.assertIsInstance(expr, BinOp)
        self.assertEqual(1, expr.eval(ns))

        expr = Expr.parse("b ** c")
        self.assertIsInstance(expr, BinOp)
        self.assertEqual(8, expr.eval(ns))

    def test_cmp_op(self):
        ns = dict(a=1, b=2, c=3)

        expr = Expr.parse("a == 1")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(True, expr.eval(ns))

        expr = Expr.parse("a != 1")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(False, expr.eval(ns))

        expr = Expr.parse("a != b != c")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(True, expr.eval(ns))

        expr = Expr.parse("a > b")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(False, expr.eval(ns))

        expr = Expr.parse("a >= 1")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(True, expr.eval(ns))

        expr = Expr.parse("a < b")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(True, expr.eval(ns))

        expr = Expr.parse("a < b < c")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(True, expr.eval(ns))

        expr = Expr.parse("c < b")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(False, expr.eval(ns))

        expr = Expr.parse("a in [1, 2, 3]")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(True, expr.eval(ns))

        expr = Expr.parse("c not in [1, 2, 3]")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(False, expr.eval(ns))

        expr = Expr.parse("a is a")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(True, expr.eval(ns))

        expr = Expr.parse("a is not 'a'")
        self.assertIsInstance(expr, Compare)
        self.assertEqual(True, expr.eval(ns))

    def test_bool_op(self):
        ns = dict(a=1, b=2, c=0)

        expr = Expr.parse("a and b")
        self.assertIsInstance(expr, Expr)
        self.assertEqual(2, expr.eval(ns))

        expr = Expr.parse("a and c and b")
        self.assertIsInstance(expr, Expr)
        self.assertEqual(0, expr.eval(ns))

        expr = Expr.parse("a or b")
        self.assertIsInstance(expr, Expr)
        self.assertEqual(1, expr.eval(ns))

        expr = Expr.parse("c or b or a")
        self.assertIsInstance(expr, Expr)
        self.assertEqual(2, expr.eval(ns))

    def test_if_exp(self):
        ns = dict(a=1, b=2, c=3)

        expr = Expr.parse("c if b else a")
        self.assertIsInstance(expr, IfExp)
        self.assertEqual(3, expr.eval(ns))

    def test_parenthesis(self):
        ns = dict(a=1, b=2, c=3)

        expr = Expr.parse("(b + a) * c")
        self.assertIsInstance(expr, Expr)
        self.assertEqual(9, expr.eval(ns))

    def test_unsupported(self):
        self.assert_unsupported("lambda x: True")
        self.assert_unsupported("await f(x)")
        self.assert_unsupported("a, b, c")

    def test_invalid(self):
        self.assert_invalid("yield True")
        self.assert_invalid("return x")
        self.assert_invalid("raise 'error!'")
        self.assert_invalid("a = 2")
        self.assert_invalid("a; b; c")

    @staticmethod
    def assert_unsupported(code: str):
        with pytest.raises(SyntaxError, match="unsupported expression"):
            Expr.parse(code)

    @staticmethod
    def assert_invalid(code: str):
        with pytest.raises(SyntaxError, match="invalid syntax"):
            Expr.parse(code)

    def test_get_safe_numpy_funcs(self):
        funcs = get_safe_numpy_funcs()
        self.assertIsInstance(funcs, dict)
        self.assertTrue(len(funcs) > 50)
        print(sorted(funcs.keys()))

    def test_get_safe_python_funcs(self):
        funcs = get_safe_python_funcs()
        self.assertIsInstance(funcs, dict)
        self.assertEqual(45, len(funcs))
        print(sorted(funcs))

    def test_performance(self):
        import timeit

        number = 10000

        ns = dict(B04=7.2, B06=0.3, a=6, b=0.2, c=0.5)
        code = "((a * B04 - b * B06) / (a * B04 + b * B06)) ** c"

        parse_timings = timeit.repeat(
            "expr = Expr.parse(code)", number=number, globals=dict(Expr=Expr, code=code)
        )

        expr = Expr.parse(code)

        eval_timings = timeit.repeat(
            "result = expr.eval(ns)", number=number, globals=dict(expr=expr, ns=ns)
        )

        sec2micros = 1000 * 1000
        parse_min_micros = round(sec2micros * min(*parse_timings) / number)
        eval_min_micros = round(sec2micros * min(*eval_timings) / number)

        print("parse_timings:", parse_timings)
        print("eval_timings:", eval_timings)
        print("parse_min_micros:", parse_min_micros)
        print("eval_min_micros:", eval_min_micros)

        self.assertTrue(parse_min_micros < 100)
        self.assertTrue(eval_min_micros < 10)
