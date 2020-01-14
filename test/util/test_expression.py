import unittest

import numpy as np
import xarray as xr

from xcube.util.expression import transpile_expr, compute_expr, compute_array_expr


class ComputeArrayExprTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_valid_exprs(self):
        namespace = dict(a=np.array([0.1, 0.3, 0.1, 0.7, 0.4, 0.9]),
                         b=np.array([0.2, 0.1, 0.3, 0.2, 0.4, 0.8]),
                         np=np,
                         xr=xr)

        value = compute_array_expr('a + 1', namespace=namespace)
        np.testing.assert_array_almost_equal(value,
                                             np.array([1.1, 1.3, 1.1, 1.7, 1.4, 1.9]))

        value = compute_array_expr('a * b', namespace=namespace)
        np.testing.assert_array_almost_equal(value,
                                             np.array([0.02, 0.03, 0.03, 0.14, 0.16, 0.72]))

        value = compute_array_expr('max(a, b)', namespace=namespace)
        np.testing.assert_array_almost_equal(value,
                                             np.array([0.2, 0.3, 0.3, 0.7, 0.4, 0.9]))

        value = compute_array_expr('a > b', namespace=namespace)
        np.testing.assert_equal(value,
                                np.array([False, True, False, True, False, True]))

        value = compute_array_expr('a == b', namespace=namespace)
        np.testing.assert_equal(value,
                                np.array([False, False, False, False, True, False]))

        # This weirdo expression is a result of translating SNAP conditional expressions to Python.
        value = compute_array_expr('a > 0.35 if a else b', namespace=namespace)
        np.testing.assert_equal(value,
                                np.array([0.2, 0.1, 0.3, 0.7, 0.4, 0.9]))

        # We actually mean
        value = compute_array_expr('where(a > 0.35, a, b)', namespace=namespace)
        np.testing.assert_equal(value,
                                np.array([0.2, 0.1, 0.3, 0.7, 0.4, 0.9]))

    def test_invalid_exprs(self):
        with self.assertRaises(ValueError) as cm:
            compute_expr('20. -')
        self.assertEqual("failed computing expression '20. -': unexpected EOF while parsing (<string>, line 1)",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            compute_expr('2 - a')
        self.assertEqual("failed computing expression '2 - a': name 'a' is not defined",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            compute_expr('b < 1', result_name="mask of 'CHL'")
        self.assertEqual("failed computing mask of 'CHL' from expression 'b < 1': name 'b' is not defined",
                         f'{cm.exception}')


class ComputeExprTest(unittest.TestCase):
    def test_valid_exprs(self):
        namespace = dict(a=3)
        self.assertEqual(2, compute_expr('2'))
        self.assertEqual("u", compute_expr('"u"'))
        self.assertEqual(3, compute_expr('a', namespace=namespace))
        self.assertEqual(True, compute_expr('True'))
        self.assertEqual(True, compute_expr('a == 3', namespace=namespace))
        self.assertEqual(5, compute_expr('a + 2', namespace=namespace))

    def test_invalid_exprs(self):
        with self.assertRaises(ValueError) as cm:
            compute_expr('20. -')
        self.assertEqual("failed computing expression '20. -': unexpected EOF while parsing (<string>, line 1)",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            compute_expr('2 - a')
        self.assertEqual("failed computing expression '2 - a': name 'a' is not defined",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            compute_expr('b < 1', result_name="mask of 'CHL'")
        self.assertEqual("failed computing mask of 'CHL' from expression 'b < 1': name 'b' is not defined",
                         f'{cm.exception}')


class TranspileExprTest(unittest.TestCase):
    def test_unary(self):
        self.assertEqual(transpile_expr('-(-+(-x))'), '--+-x')
        self.assertEqual(transpile_expr('-((a-b)-c)'), '-(a - b - c)')
        self.assertEqual(transpile_expr('not x'), 'np.logical_not(x)')
        self.assertEqual(transpile_expr('not -x'), 'np.logical_not(-x)')
        self.assertEqual(transpile_expr('not not x'), 'np.logical_not(np.logical_not(x))')

    def test_binary(self):
        self.assertEqual(transpile_expr('a-b-c-d'), 'a - b - c - d')
        self.assertEqual(transpile_expr('(a-b)-c-d'), 'a - b - c - d')
        self.assertEqual(transpile_expr('(a-b-c)-d'), 'a - b - c - d')
        self.assertEqual(transpile_expr('a-(b-c)-d'), 'a - (b - c) - d')
        self.assertEqual(transpile_expr('a-(b-c-d)'), 'a - (b - c - d)')
        self.assertEqual(transpile_expr('a-b-(c-d)'), 'a - b - (c - d)')
        self.assertEqual(transpile_expr('a-(b-(c-d))'), 'a - (b - (c - d))')
        self.assertEqual(transpile_expr('(a-(b-c))-d'), 'a - (b - c) - d')

        self.assertEqual(transpile_expr('a----b'), 'a - ---b')
        self.assertEqual(transpile_expr('---a-b'), '---a - b')

        self.assertEqual(transpile_expr('a*b+c/d'), 'a * b + c / d')
        self.assertEqual(transpile_expr('a+b*c-d'), 'a + b * c - d')
        self.assertEqual(transpile_expr('(a+b)*(c-d)'), '(a + b) * (c - d)')

        self.assertEqual(transpile_expr('a ** b'), 'np.power(a, b)')

    def test_bool(self):
        self.assertEqual(transpile_expr('a and b and c'), 'np.logical_and(np.logical_and(a, b), c)')
        self.assertEqual(transpile_expr('(a and b) and c'), 'np.logical_and(np.logical_and(a, b), c)')
        self.assertEqual(transpile_expr('a and (b and c)'), 'np.logical_and(a, np.logical_and(b, c))')
        self.assertEqual(transpile_expr('(a or b) and not (c or not d)'),
                         'np.logical_and(np.logical_or(a, b), np.logical_not(np.logical_or(c, np.logical_not(d))))')

        self.assertEqual(transpile_expr('a and b or c and d'),
                         'np.logical_or(np.logical_and(a, b), np.logical_and(c, d))')
        self.assertEqual(transpile_expr('a or b and c or d'),
                         'np.logical_or(np.logical_or(a, np.logical_and(b, c)), d)')
        self.assertEqual(transpile_expr('(a or b) and (c or d)'),
                         'np.logical_and(np.logical_or(a, b), np.logical_or(c, d))')

    def test_compare(self):
        self.assertEqual(transpile_expr('a < 2'), 'a < 2')
        self.assertEqual(transpile_expr('a != 2'), 'a != 2')
        self.assertEqual(transpile_expr('a == NaN'), 'np.isnan(a)')
        self.assertEqual(transpile_expr('a != NaN'), 'np.logical_not(np.isnan(a))')
        self.assertEqual(transpile_expr('a is not Null'), 'a is not Null')
        self.assertEqual(transpile_expr('a in data'), 'a in data')

        with self.assertRaises(ValueError) as cm:
            transpile_expr('a >= 2 == b')
        self.assertEqual(str(cm.exception),
                         'expression "a >= 2 == b" uses an n-ary comparison, but only binary are supported')

        with self.assertRaises(ValueError) as cm:
            transpile_expr('0 < x <= 1')
        self.assertEqual(str(cm.exception),
                         'expression "0 < x <= 1" uses an n-ary comparison, but only binary are supported')

    def test_attributes(self):
        self.assertEqual(transpile_expr('a.b'), 'a.b')

    def test_functions(self):
        self.assertEqual(transpile_expr('sin(x)'), 'np.sin(x)')
        self.assertEqual(transpile_expr('min(x, y)'), 'np.fmin(x, y)')
        self.assertEqual(transpile_expr('max(x, y)'), 'np.fmax(x, y)')
        self.assertEqual(transpile_expr('isnan(x, y)'), 'np.isnan(x, y)')

    def test_conditional(self):
        # The following conditional expr looks wrong but this is how it looks like after translating
        # from SNAP expression 'a >= 0.0 ? a : NaN'
        self.assertEqual(transpile_expr('a >= 0.0 if a else NaN'),
                         'xr.where(a >= 0.0, a, NaN)')
        self.assertEqual(transpile_expr('a >= 0.0 if a else b >= 0.0 if b else NaN'),
                         'xr.where(a >= 0.0, a, xr.where(b >= 0.0, b, NaN))')

    def test_where(self):
        self.assertEqual(transpile_expr('where(a >= 0.0, a, NaN)'),
                         'xr.where(a >= 0.0, a, NaN)')
        self.assertEqual(transpile_expr('xr.where(a >= 0.0, a, NaN)'),
                         'xr.where(a >= 0.0, a, NaN)')
        self.assertEqual(transpile_expr('np.where(a >= 0.0, a, NaN)'),
                         'np.where(a >= 0.0, a, NaN)')

    def test_mixed(self):
        self.assertEqual(transpile_expr('a+sin(x + 2.8)'),
                         'a + np.sin(x + 2.8)')
        self.assertEqual(transpile_expr('a+max(1, sin(x+2.8), x**0.5)'),
                         'a + np.fmax(1, np.sin(x + 2.8), np.power(x, 0.5))')
