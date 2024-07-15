# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import math
import unittest

import pytest

from xcube.core.varexpr import VarExprError
from xcube.core.varexpr import evaluate


class VarExprEvaluateTest(unittest.TestCase):
    def test_constant(self):
        names = {}
        self.assertEqual(None, evaluate("None", names))
        self.assertEqual(True, evaluate("True", names))
        self.assertEqual(False, evaluate("False", names))
        self.assertEqual(13, evaluate("13", names))
        self.assertEqual(0.25, evaluate("0.25", names))
        self.assertEqual("TEST", evaluate("'TEST'", names))

    def test_name(self):
        names = {"A": 137, "B": 0.5}
        self.assertEqual(137, evaluate("A", names))
        self.assertEqual(0.5, evaluate("B", names))

    def test_attribute(self):
        class A:
            pass

        a = A()
        a.x = 13
        a._x = 14

        names = {"A": a}

        self.assertEqual(13, evaluate("A.x", names))

        with pytest.raises(VarExprError, match="'A' object has no attribute 'y'"):
            evaluate("A.y", names)

        with pytest.raises(
            VarExprError, match="illegal use of protected attribute '_x'"
        ):
            evaluate("A._x", names)

        with pytest.raises(
            VarExprError, match="illegal use of protected attribute '__class__'"
        ):
            evaluate("A.__class__", names)

    def test_subscript(self):
        names = {"A": [10, 11, 12]}
        self.assertEqual(10, evaluate("A[0]", names))
        self.assertEqual(11, evaluate("A[1]", names))
        self.assertEqual(12, evaluate("A[2]", names))
        self.assertEqual(12, evaluate("A[-1]", names))

    def test_slice(self):
        names = {"A": [10, 11, 12, 13, 14]}
        self.assertEqual(names["A"], evaluate("A[:]", names))
        self.assertEqual([11, 12], evaluate("A[1:3]", names))
        self.assertEqual([10, 12, 14], evaluate("A[0:6:2]", names))
        self.assertEqual([12, 13, 14], evaluate("A[2:]", names))
        self.assertEqual([10, 11], evaluate("A[:2]", names))

    def test_call(self):
        names = {"sin": math.sin, "sqrt": math.sqrt}
        self.assertEqual(0, evaluate("sin(0)", names))
        self.assertEqual(4, evaluate("sqrt(16)", names))

        def poly(x, a0=0, a1=0, a2=0):
            return a0 + x * (a1 + x * a2)

        names = {"poly": poly}
        self.assertEqual(3, evaluate("poly(2, a0=3)", names))
        self.assertEqual(5, evaluate("poly(2, a0=3, a1=1)", names))
        self.assertEqual(13, evaluate("poly(2, a0=3, a1=1, a2=2)", names))

    def test_unary(self):
        names = {"A": 255, "B": False}
        self.assertEqual(255, evaluate("+A", names))
        self.assertEqual(0, evaluate("+B", names))
        self.assertEqual(-255, evaluate("-A", names))
        self.assertEqual(0, evaluate("-B", names))
        self.assertEqual(-256, evaluate("~A", names))
        self.assertEqual(-1, evaluate("~B", names))
        self.assertEqual(False, evaluate("not A", names))
        self.assertEqual(True, evaluate("not B", names))

    def test_binary(self):
        names = {"A": 15, "B": 7}
        self.assertEqual(22, evaluate("A + B", names))
        self.assertEqual(8, evaluate("A - B", names))
        self.assertEqual(105, evaluate("A * B", names))
        self.assertEqual(15 / 7, evaluate("A / B", names))
        self.assertEqual(2, evaluate("A // B", names))
        self.assertEqual(1, evaluate("A % B", names))
        self.assertEqual(3375, evaluate("A ** 3", names))
        self.assertEqual(120, evaluate("A << 3", names))
        self.assertEqual(3, evaluate("A >> 2", names))
        self.assertEqual(7, evaluate("A & B", names))
        self.assertEqual(8, evaluate("A ^ B", names))
        self.assertEqual(15, evaluate("A | B", names))

    def test_comparison(self):
        names = {"A": 15, "B": 7}
        self.assertEqual(True, evaluate("A == A", names))
        self.assertEqual(False, evaluate("A == B", names))
        self.assertEqual(False, evaluate("A != A", names))
        self.assertEqual(True, evaluate("A != B", names))
        self.assertEqual(False, evaluate("A < A", names))
        self.assertEqual(False, evaluate("A < B", names))
        self.assertEqual(True, evaluate("A <= A", names))
        self.assertEqual(False, evaluate("A <= B", names))
        self.assertEqual(False, evaluate("A > A", names))
        self.assertEqual(True, evaluate("A > B", names))
        self.assertEqual(True, evaluate("A >= A", names))
        self.assertEqual(True, evaluate("A >= B", names))

        self.assertEqual(True, evaluate("10 < A < 20", names))
        self.assertEqual(False, evaluate("10 > A > 20", names))
        self.assertEqual(True, evaluate("10 > B < 20", names))
        self.assertEqual(False, evaluate("10 < B < 20", names))

        self.assertEqual(True, evaluate("'c' in 'abc'", names))
        self.assertEqual(False, evaluate("'d' in 'abc'", names))
        self.assertEqual(False, evaluate("'c' not in 'abc'", names))
        self.assertEqual(True, evaluate("'d' not in 'abc'", names))

        self.assertEqual(True, evaluate("A is A", names))
        self.assertEqual(False, evaluate("A is B", names))
        self.assertEqual(False, evaluate("A is not A", names))
        self.assertEqual(True, evaluate("A is not B", names))

    def test_bool_op(self):
        names = {"A": 15, "B": 7, "C": 0, "D": False}

        self.assertEqual(7, evaluate("A and B", names))
        self.assertEqual(0, evaluate("A and B and C", names))
        self.assertEqual(0, evaluate("A and B and C and D", names))
        self.assertEqual(False, evaluate("D and C and B and A", names))
        self.assertEqual(0, evaluate("C and B and A", names))
        self.assertEqual(15, evaluate("B and A", names))

        self.assertEqual(15, evaluate("A or B", names))
        self.assertEqual(15, evaluate("A or B or C", names))
        self.assertEqual(15, evaluate("A or B or C or D", names))
        self.assertEqual(7, evaluate("D or C or B or A", names))
        self.assertEqual(7, evaluate("C or B or A", names))
        self.assertEqual(7, evaluate("B or A", names))

    def test_if(self):
        names = {"A": 15, "B": 7, "C": True, "D": False}
        self.assertEqual(15, evaluate("A if C else B", names))
        self.assertEqual(7, evaluate("A if D else B", names))

    # noinspection PyMethodMayBeStatic
    def test_lambda(self):
        names = {"A": 15, "B": 7, "C": 2}
        with pytest.raises(
            VarExprError, match="unsupported expression node of type 'Lambda'"
        ):
            evaluate("lambda x: A * B + C", names)
