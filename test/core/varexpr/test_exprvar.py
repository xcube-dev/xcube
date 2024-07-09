# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Any

import pytest
import xarray as xr

from xcube.core.varexpr.exprvar import ExprVar


class ExprVarTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_that_ctor_raises(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            ExprVar(137)

    def test_that_all_xarray_ops_are_supported(self):
        da1 = xr.DataArray([1, 2, 3], dims="x")
        da2 = xr.DataArray([2, 3, 4], dims="x")
        ev1 = ExprVar(da1)
        ev2 = ExprVar(da2)

        # Binary operations - comparisons

        self.assert_ev(ev1 == ev2, [False, False, False])
        self.assert_ev(ev1 == 1, [True, False, False])

        self.assert_ev(ev1 != ev2, [True, True, True])
        self.assert_ev(ev1 != 1, [False, True, True])

        self.assert_ev(ev1 <= ev2, [True, True, True])
        self.assert_ev(ev1 <= 1, [True, False, False])

        self.assert_ev(ev1 < ev2, [True, True, True])
        self.assert_ev(ev1 < 1, [False, False, False])

        self.assert_ev(ev1 >= ev2, [False, False, False])
        self.assert_ev(ev1 >= 1, [True, True, True])

        self.assert_ev(ev1 > ev2, [False, False, False])
        self.assert_ev(ev1 > 1, [False, True, True])

        # Binary operations - emulating numeric type

        self.assert_ev(ev1 + ev2, [3, 5, 7])
        self.assert_ev(ev1 + 3, [4, 5, 6])
        self.assert_ev(1 + ev2, [3, 4, 5])

        self.assert_ev(ev1 - ev2, [-1, -1, -1])
        self.assert_ev(ev1 - 1, [0, 1, 2])
        self.assert_ev(5 - ev2, [3, 2, 1])

        self.assert_ev(ev1 * ev2, [2, 6, 12])
        self.assert_ev(ev1 * 2, [2, 4, 6])
        self.assert_ev(3 * ev2, [6, 9, 12])

        self.assert_ev(ev2 / ev1, [2.0, 1.5, 4 / 3])
        self.assert_ev(ev1 / 2, [0.5, 1.0, 1.5])
        self.assert_ev(3 / ev2, [1.5, 1.0, 0.75])

        self.assert_ev(ev2 // ev1, [2, 1, 1])
        self.assert_ev(ev1 // 2, [0, 1, 1])
        self.assert_ev(3 // ev2, [1, 1, 0])

        self.assert_ev(ev2 % ev1, [0, 1, 1])
        self.assert_ev(ev1 % 2, [1, 0, 1])
        self.assert_ev(3 % ev2, [1, 0, 3])

        self.assert_ev(ev1**ev2, [1, 8, 81])
        self.assert_ev(ev1**3, [1, 8, 27])
        self.assert_ev(2**ev2, [4, 8, 16])

        self.assert_ev(ev1 << ev2, [4, 16, 48])
        self.assert_ev(ev1 << 1, [2, 4, 6])

        self.assert_ev(ev1 >> ev2, [0, 0, 0])
        self.assert_ev(ev1 >> 1, [0, 1, 1])

        self.assert_ev(ev1 & ev2, [0, 2, 0])
        self.assert_ev(ev1 & 3, [1, 2, 3])
        self.assert_ev(2 & ev2, [2, 2, 0])

        self.assert_ev(ev1 ^ ev2, [3, 1, 7])
        self.assert_ev(ev1 ^ 3, [2, 1, 0])
        self.assert_ev(2 ^ ev2, [0, 1, 6])

        self.assert_ev(ev1 | ev2, [3, 3, 7])
        self.assert_ev(ev1 | 3, [3, 3, 3])
        self.assert_ev(2 | ev2, [2, 3, 6])

        # Unary operations

        self.assert_ev(+ev1, [1, 2, 3])
        self.assert_ev(-ev1, [-1, -2, -3])
        self.assert_ev(~ev1, [-2, -3, -4])

        # Unary abs()
        self.assert_ev(abs(ev1), [1, 2, 3])
        self.assert_ev(abs(-ev1), [1, 2, 3])

    def assert_ev(self, ev: Any, expected_result: list):
        self.assertIsInstance(ev, ExprVar)
        da = ev.__dict__["_ExprVar__da"]
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(expected_result, list(da.values))

    # noinspection PyMethodMayBeStatic
    def test_that_data_array_is_not_easily_accessible(self):
        da = xr.DataArray([1, 2, 3], dims="x")
        ev = ExprVar(da)

        with pytest.raises(
            AttributeError,
            match="'ExprVar' object has no attribute '_ExprVarTest__da'",
        ):
            # noinspection PyUnusedLocal,PyUnresolvedReferences
            result = ev.__da

    # noinspection PyMethodMayBeStatic
    def test_that_some_ops_are_unsupported(self):
        ev = ExprVar(xr.DataArray([1, 2, 3], dims="x"))

        with pytest.raises(
            TypeError,
            match="unsupported operand type\\(s\\) for <<: 'int' and 'DataArray'",
        ):
            # noinspection PyUnusedLocal
            result = 1 << ev

        with pytest.raises(
            TypeError,
            match="unsupported operand type\\(s\\) for >>: 'int' and 'DataArray'",
        ):
            # noinspection PyUnusedLocal
            result = 1 >> ev
