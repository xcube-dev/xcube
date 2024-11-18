# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import pytest
import xarray as xr
import numpy as np

from xcube.core.varexpr import VarExprContext
from xcube.core.varexpr import VarExprError
from xcube.core.varexpr.exprvar import ExprVar

var_a_data = np.array([[0, 1], [2, 3]])
var_b_data = np.array([[1, 2], [3, 4]])
var_c_data = np.array([[1, 0], [0, 1]])
dataset = xr.Dataset(
    dict(
        A=xr.DataArray(var_a_data, dims=("y", "x")),
        B=xr.DataArray(var_b_data, dims=("y", "x")),
        C=xr.DataArray(
            var_c_data,
            dims=("y", "x"),
            attrs={"flag_meanings": "on off", "flag_values": "0, 1"},
        ),
    )
)

# https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
# Tested with Python 3.12: it works, Python interpreter will crash
# for eval(BOMB).
BOMB = """
(lambda fc=(
    lambda n: [
        c for c in
            ().__class__.__bases__[0].__subclasses__()
            if c.__name__ == n
        ][0]
    ):
    fc("function")(
        fc("code")(
            # 2.7:          0,0,0,0,"BOOM",(),(),(),"","",0,""
            # 3.5-3.7:      0,0,0,0,0,b"BOOM",(),(),(),"","",0,b""
            # 3.8-3.10:     0,0,0,0,0,0,b"BOOM",(),(),(),"","",0,b""
            # 3.11:         
            0,0,0,0,0,0,b"BOOM",(),(),(),"","","",0,b"",b"",(),()
        ),{}
    )()
)()
"""


class VarExprContextTest(unittest.TestCase):
    def test_names(self):
        ctx = VarExprContext(dataset)
        names = ctx.names()

        self.assertIsInstance(names.get("A"), ExprVar)
        self.assertIsInstance(names.get("B"), ExprVar)
        self.assertIsInstance(names.get("C"), ExprVar)

        self.assertIsInstance(names.get("nan"), float)
        self.assertIsInstance(names.get("inf"), float)
        self.assertIsInstance(names.get("e"), float)
        self.assertIsInstance(names.get("pi"), float)

        self.assertTrue(callable(names.get("where")))
        self.assertTrue(callable(names.get("hypot")))
        self.assertTrue(callable(names.get("sin")))
        self.assertTrue(callable(names.get("cos")))
        self.assertTrue(callable(names.get("tan")))
        self.assertTrue(callable(names.get("sqrt")))
        self.assertTrue(callable(names.get("fmin")))
        self.assertTrue(callable(names.get("fmax")))
        self.assertTrue(callable(names.get("minimum")))
        self.assertTrue(callable(names.get("maximum")))

        self.assertNotIn("min", names)
        self.assertNotIn("max", names)
        self.assertNotIn("open", names)

    def test_capabilities(self):
        c_list = VarExprContext.get_constants()
        self.assertIsInstance(c_list, list)
        # print(c_list)

        fn_list = VarExprContext.get_array_functions()
        self.assertIsInstance(fn_list, list)
        # print(fn_list)
        fn_list = VarExprContext.get_other_functions()
        self.assertIsInstance(fn_list, list)
        # print(fn_list)

        fn_list = VarExprContext.get_array_operators()
        self.assertIsInstance(fn_list, list)
        # print(fn_list)
        fn_list = VarExprContext.get_other_operators()
        self.assertIsInstance(fn_list, list)
        # print(fn_list)

    def test_evaluate(self):
        ctx = VarExprContext(dataset)

        # Arithmetic
        result = ctx.evaluate("A + B")
        self.assertIsInstance(result, xr.DataArray)
        np.testing.assert_equal(result.values, var_a_data + var_b_data)

        # Asserting that one numpy ufunc works should be ok
        result = ctx.evaluate("hypot(A, B)")
        self.assertIsInstance(result, xr.DataArray)
        np.testing.assert_equal(
            result.values,
            np.hypot(var_a_data, var_b_data),
        )

        # Comparison and bitwise ops
        result = ctx.evaluate("(A <= 2) & (B >= 3)")
        self.assertIsInstance(result, xr.DataArray)
        np.testing.assert_equal(result.values, (var_a_data <= 2) & (var_b_data >= 3))

    def test_syntax_errors(self):
        ctx = VarExprContext(dataset)
        self.assert_syntax_error(ctx, "x = A")
        self.assert_syntax_error(ctx, "A + 1; B + 2")

    # noinspection PyMethodMayBeStatic
    @staticmethod
    def assert_syntax_error(ctx: VarExprContext, expr: str):
        with pytest.raises(VarExprError, match="invalid syntax"):
            ctx.evaluate(expr)

    # noinspection PyMethodMayBeStatic
    def test_that_harmless_builtins_are_callable(self):
        ctx = VarExprContext(dataset)

        result = ctx.evaluate("abs(B)")
        self.assertIsInstance(result, xr.DataArray)

        result = ctx.evaluate("floor(B)")
        self.assertIsInstance(result, xr.DataArray)

        result = ctx.evaluate("ceil(B)")
        self.assertIsInstance(result, xr.DataArray)

    # noinspection PyMethodMayBeStatic
    def test_that_confusing_builtins_are_not_callable(self):
        ctx = VarExprContext(dataset)

        with pytest.raises(
            VarExprError,
            match="name 'min' is not defined",
        ):
            ctx.evaluate("min(A, B)")

        with pytest.raises(
            VarExprError,
            match="name 'max' is not defined",
        ):
            ctx.evaluate("max(A, B)")

    # noinspection PyMethodMayBeStatic
    def test_that_dangerous_builtins_are_not_callable(self):
        ctx = VarExprContext(dataset)

        with pytest.raises(
            VarExprError,
            match="name 'open' is not defined",
        ):
            ctx.evaluate("open('mycode.py', 'w')")

        with pytest.raises(
            VarExprError,
            match="name 'input' is not defined",
        ):
            ctx.evaluate("input()")

        with pytest.raises(
            VarExprError,
            match="name '__module__' is not defined",
        ):
            ctx.evaluate("__module__")

        with pytest.raises(
            VarExprError,
            match="name '__import__' is not defined",
        ):
            ctx.evaluate("__import__('evilmodule')")

    # noinspection PyMethodMayBeStatic
    def test_that_lambda_is_not_supported(self):
        ctx = VarExprContext(dataset)
        with pytest.raises(
            VarExprError,
            match="unsupported expression node of type 'Lambda'",
        ):
            ctx.evaluate(BOMB)

    # noinspection PyMethodMayBeStatic
    def test_invalid_expression_cases(self):
        ctx = VarExprContext(dataset)

        with pytest.raises(
            VarExprError,
            match="name 'my_secret_var' is not defined",
        ):
            ctx.evaluate("my_secret_var")

        with pytest.raises(
            VarExprError,
            match="'DataArray' object has no attribute 'values'",
        ):
            ctx.evaluate("A.values")

        with pytest.raises(
            VarExprError,
            match="'DataArray' object is not subscriptable",
        ):
            ctx.evaluate("A[0, 0]")

        with pytest.raises(
            VarExprError,
            match="result must be a 'DataArray' object, but got type 'int'",
        ):
            ctx.evaluate("137")

        with pytest.raises(
            VarExprError,
            match="result must be a 'DataArray' object, but got type 'tuple'",
        ):
            ctx.evaluate("A, B")

    # noinspection PyMethodMayBeStatic
    def test_intentional_disintegration(self):
        ctx = VarExprContext(dataset)
        # This is intentional disintegration
        ev = ExprVar(xr.DataArray(var_a_data, dims=("y", "x")))
        ev.__dict__["_ExprVar__da"] = True
        ctx._names["my_ev"] = ev
        with pytest.raises(
            RuntimeError,
            match="internal error: 'DataArray' object expected, but got type 'bool'",
        ):
            ctx.evaluate("my_ev")
