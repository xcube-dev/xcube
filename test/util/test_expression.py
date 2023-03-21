import unittest
from collections import namedtuple

import numpy as np
import numpy.testing as npt
import xarray as xr

from xcube.util.expression import compute_array_expr
from xcube.util.expression import compute_expr
from xcube.util.expression import transpile_expr


class ComputeArrayExprTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_valid_exprs(self):
        namespace = dict(a=np.array([0.1, 0.3, 0.1, 0.7, 0.4, 0.9]),
                         b=np.array([0.2, 0.1, 0.3, 0.2, 0.4, 0.8]),
                         np=np,
                         xr=xr)

        value = compute_array_expr('a + 1', namespace=namespace)
        npt.assert_array_almost_equal(value,
                                      np.array([1.1, 1.3, 1.1, 1.7, 1.4, 1.9]))

        value = compute_array_expr('a * b', namespace=namespace)
        npt.assert_array_almost_equal(value,
                                      np.array([0.02, 0.03, 0.03, 0.14, 0.16, 0.72]))

        value = compute_array_expr('max(a, b)', namespace=namespace)
        npt.assert_array_almost_equal(value,
                                      np.array([0.2, 0.3, 0.3, 0.7, 0.4, 0.9]))

        value = compute_array_expr('a > b', namespace=namespace)
        npt.assert_equal(value,
                         np.array([False, True, False, True, False, True]))

        value = compute_array_expr('a == b', namespace=namespace)
        npt.assert_equal(value,
                         np.array([False, False, False, False, True, False]))

        # This weirdo expression is a result of translating SNAP conditional expressions to Python.
        value = compute_array_expr('a > 0.35 if a else b', namespace=namespace)
        npt.assert_equal(value,
                         np.array([0.2, 0.1, 0.3, 0.7, 0.4, 0.9]))

        # We actually mean
        value = compute_array_expr('where(a > 0.35, a, b)', namespace=namespace)
        npt.assert_equal(value,
                         np.array([0.2, 0.1, 0.3, 0.7, 0.4, 0.9]))

    def test_invalid_exprs(self):
        with self.assertRaises(ValueError) as cm:
            compute_expr('20. -')
        self.assertTrue(f'{cm.exception}'.startswith(
            "failed computing expression '20. -': ")
        )

        with self.assertRaises(ValueError) as cm:
            compute_expr('2 - a')
        self.assertTrue(f'{cm.exception}'.startswith(
            "failed computing expression '2 - a': "
        ))

        with self.assertRaises(ValueError) as cm:
            compute_expr('b < 1', result_name="mask of 'CHL'")
        self.assertTrue(f'{cm.exception}'.startswith(
            "failed computing mask of 'CHL' from expression 'b < 1': "
        ))

    # noinspection PyMethodMayBeStatic
    def test_complex_case(self):
        expr = ('(not quality_flags.invalid'
                ' and not pixel_classif_flags.IDEPIX_CLOUD'
                ' and not pixel_classif_flags.IDEPIX_CLOUD_BUFFER'
                ' and not pixel_classif_flags.IDEPIX_CLOUD_SHADOW'
                ' and not pixel_classif_flags.IDEPIX_SNOW_ICE'
                ' and not (c2rcc_flags.Rtosa_OOS and conc_chl > 1.0)'
                ' and not c2rcc_flags.Rtosa_OOR'
                ' and not c2rcc_flags.Rhow_OOR'
                ' and not (c2rcc_flags.Cloud_risk and immersed_cyanobacteria == 0)'
                ' and floating_vegetation == 0'
                ' and conc_chl > 0.01'
                ' and not (floating_cyanobacteria == 1 or chl_pitarch > 500))')

        quality_flags = namedtuple('quality_flags',
                                   ['invalid'])
        quality_flags.invalid = np.array([0])
        pixel_classif_flags = namedtuple('pixel_classif_flags',
                                         ['IDEPIX_CLOUD',
                                          'IDEPIX_CLOUD_BUFFER',
                                          'IDEPIX_CLOUD_SHADOW',
                                          'IDEPIX_SNOW_ICE'])
        pixel_classif_flags.IDEPIX_CLOUD = np.array([0])
        pixel_classif_flags.IDEPIX_CLOUD_BUFFER = np.array([0])
        pixel_classif_flags.IDEPIX_CLOUD_SHADOW = np.array([0])
        pixel_classif_flags.IDEPIX_SNOW_ICE = np.array([0])
        c2rcc_flags = namedtuple('c2rcc_flags',
                                 ['Rtosa_OOS',
                                  'Rtosa_OOR',
                                  'Rhow_OOR',
                                  'Cloud_risk'])
        c2rcc_flags.Rtosa_OOS = np.array([0])
        c2rcc_flags.Rtosa_OOR = np.array([0])
        c2rcc_flags.Rhow_OOR = np.array([0])
        c2rcc_flags.Cloud_risk = np.array([0])
        namespace = dict(
            np=np,
            quality_flags=quality_flags,
            pixel_classif_flags=pixel_classif_flags,
            c2rcc_flags=c2rcc_flags,
            immersed_cyanobacteria=np.array([0]),
            floating_cyanobacteria=np.array([0]),
            floating_vegetation=np.array([0]),
            conc_chl=np.array([0]),
            chl_pitarch=np.array([0]),
        )

        actual_value = compute_array_expr(expr, namespace=namespace)
        expected_value = 0
        npt.assert_array_almost_equal(actual_value, np.array([expected_value]))

        namespace['conc_chl'] = np.array([0.2])
        actual_value = compute_array_expr(expr, namespace=namespace)
        expected_value = 1
        npt.assert_array_almost_equal(actual_value, np.array([expected_value]))

        pixel_classif_flags.IDEPIX_CLOUD_SHADOW = np.array([0.2])
        actual_value = compute_array_expr(expr, namespace=namespace)
        expected_value = 0
        npt.assert_array_almost_equal(actual_value, np.array([expected_value]))


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
        self.assertTrue(f'{cm.exception}'.startswith(
            "failed computing expression '20. -': "
        ))

        with self.assertRaises(ValueError) as cm:
            compute_expr('2 - a')
        self.assertTrue(f'{cm.exception}'.startswith(
            "failed computing expression '2 - a': "
        ))

        with self.assertRaises(ValueError) as cm:
            compute_expr('b < 1', result_name="mask of 'CHL'")
        self.assertTrue(f'{cm.exception}'.startswith(
            "failed computing mask of 'CHL' from "
            "expression 'b < 1': "
        ))


class TranspileExprTest(unittest.TestCase):
    def test_unary(self):
        self.assertEqual('--+-x',
                         transpile_expr('-(-+(-x))'))
        self.assertEqual('-(a - b - c)',
                         transpile_expr('-((a-b)-c)'))
        self.assertEqual('np.logical_not(x)',
                         transpile_expr('not x'))
        self.assertEqual('np.logical_not(-x)',
                         transpile_expr('not -x'))
        self.assertEqual('np.logical_not(np.logical_not(x))',
                         transpile_expr('not not x'))

    def test_binary(self):
        self.assertEqual('a - b - c - d',
                         transpile_expr('a-b-c-d'))
        self.assertEqual('a - b - c - d',
                         transpile_expr('(a-b)-c-d'))
        self.assertEqual('a - b - c - d',
                         transpile_expr('(a-b-c)-d'))
        self.assertEqual('a - (b - c) - d',
                         transpile_expr('a-(b-c)-d'))
        self.assertEqual('a - (b - c - d)',
                         transpile_expr('a-(b-c-d)'))
        self.assertEqual('a - b - (c - d)',
                         transpile_expr('a-b-(c-d)'))
        self.assertEqual('a - (b - (c - d))',
                         transpile_expr('a-(b-(c-d))'))
        self.assertEqual('a - (b - c) - d',
                         transpile_expr('(a-(b-c))-d'))

        self.assertEqual('a - ---b',
                         transpile_expr('a----b'))
        self.assertEqual('---a - b',
                         transpile_expr('---a-b'))

        self.assertEqual('a * b + c / d',
                         transpile_expr('a*b+c/d'))
        self.assertEqual('a + b * c - d',
                         transpile_expr('a+b*c-d'))
        self.assertEqual('(a + b) * (c - d)',
                         transpile_expr('(a+b)*(c-d)'))

        self.assertEqual('np.power(a, b)',
                         transpile_expr('a ** b'))

    def test_bool(self):
        self.assertEqual('np.logical_and(np.logical_and(a, b), c)',
                         transpile_expr('a and b and c'))
        self.assertEqual('np.logical_and(np.logical_and(a, b), c)',
                         transpile_expr('(a and b) and c'))
        self.assertEqual('np.logical_and(a, np.logical_and(b, c))',
                         transpile_expr('a and (b and c)'))
        self.assertEqual('np.logical_and(np.logical_or(a, b), '
                         'np.logical_not(np.logical_or(c, '
                         'np.logical_not(d))))',
                         transpile_expr('(a or b) and not (c or not d)'))

        self.assertEqual('np.logical_or(np.logical_and(a, b), '
                         'np.logical_and(c, d))',
                         transpile_expr('a and b or c and d'))
        self.assertEqual('np.logical_or(np.logical_or(a, '
                         'np.logical_and(b, c)), d)',
                         transpile_expr('a or b and c or d'))
        self.assertEqual('np.logical_and(np.logical_or(a, b), '
                         'np.logical_or(c, d))',
                         transpile_expr('(a or b) and (c or d)'))

    def test_compare(self):
        self.assertEqual('a < 2',
                         transpile_expr('a < 2'))
        self.assertEqual('a != 2',
                         transpile_expr('a != 2'))
        self.assertEqual('np.isnan(a)',
                         transpile_expr('a == NaN'))
        self.assertEqual('np.logical_not(np.isnan(a))',
                         transpile_expr('a != NaN'))
        self.assertEqual('np.isnan(a)',
                         transpile_expr('NaN == a'))
        self.assertEqual('np.logical_not(np.isnan(a))',
                         transpile_expr('NaN != a'))
        self.assertEqual('a is not Null',
                         transpile_expr('a is not Null'))
        self.assertEqual('a in data',
                         transpile_expr('a in data'))

        with self.assertRaises(ValueError) as cm:
            transpile_expr('a >= 2 == b')
        x8 = str(cm.exception)
        self.assertEqual('expression "a >= 2 == b" uses an n-ary '
                         'comparison, but only binary are supported', x8)

        with self.assertRaises(ValueError) as cm:
            transpile_expr('0 < x <= 1')
        x9 = str(cm.exception)
        self.assertEqual('expression "0 < x <= 1" uses an n-ary '
                         'comparison, but only binary are supported', x9)

    def test_attributes(self):
        self.assertEqual('a.b',
                         transpile_expr('a.b'))

    def test_functions(self):
        self.assertEqual('np.sin(x)',
                         transpile_expr('sin(x)'))
        self.assertEqual('np.fmin(x, y)',
                         transpile_expr('min(x, y)'))
        self.assertEqual('np.fmax(x, y)',
                         transpile_expr('max(x, y)'))
        self.assertEqual('np.isnan(x, y)',
                         transpile_expr('isnan(x, y)'))

    def test_conditional(self):
        # The following conditional expr looks wrong but this 
        # is how it looks like after translating
        # from SNAP expression 'a >= 0.0 ? a : NaN'
        self.assertEqual('xr.where(a >= 0.0, a, NaN)',
                         transpile_expr('a >= 0.0 if a else NaN'))
        self.assertEqual('xr.where(a >= 0.0, a, xr.where(b >= 0.0, b, NaN))',
                         transpile_expr('a >= 0.0 if a else b >= 0.0 if b else NaN'))

    def test_where(self):
        self.assertEqual('xr.where(a >= 0.0, a, NaN)',
                         transpile_expr('where(a >= 0.0, a, NaN)'))
        self.assertEqual('xr.where(a >= 0.0, a, NaN)',
                         transpile_expr('xr.where(a >= 0.0, a, NaN)'))
        self.assertEqual('np.where(a >= 0.0, a, NaN)',
                         transpile_expr('np.where(a >= 0.0, a, NaN)'))
        # xarray.DataArray.where() method:
        self.assertEqual('a.where(a.x >= 0.0)',
                         transpile_expr('a.where(a.x >= 0.0)'))

    def test_mixed(self):
        self.assertEqual('a + np.sin(x + 2.8)',
                         transpile_expr('a+sin(x + 2.8)'))
        self.assertEqual('a + np.fmax(1, np.sin(x + 2.8), np.power(x, 0.5))',
                         transpile_expr('a+max(1, sin(x+2.8), x**0.5)'))

        expr = ('(not quality_flags.invalid'
                ' and not pixel_classif_flags.IDEPIX_CLOUD'
                ' and not pixel_classif_flags.IDEPIX_CLOUD_BUFFER'
                ' and not pixel_classif_flags.IDEPIX_CLOUD_SHADOW'
                ' and not pixel_classif_flags.IDEPIX_SNOW_ICE'
                ' and not (c2rcc_flags.Rtosa_OOS and conc_chl > 1.0)'
                ' and not c2rcc_flags.Rtosa_OOR'
                ' and not c2rcc_flags.Rhow_OOR'
                ' and not (c2rcc_flags.Cloud_risk and immersed_cyanobacteria == 0)'
                ' and floating_vegetation == 0'
                ' and conc_chl > 0.01'
                ' and not (floating_cyanobacteria == 1 or chl_pitarch > 500))')

        self.assertEqual('np.logical_and(np.logical_and('
                         'np.logical_and(np.logical_and('
                         'np.logical_and(np.logical_and('
                         'np.logical_and(np.logical_and('
                         'np.logical_and(np.logical_and('
                         'np.logical_and('
                         'np.logical_not(quality_flags.invalid), '
                         'np.logical_not(pixel_classif_flags.IDEPIX_CLOUD)), '
                         'np.logical_not(pixel_classif_flags.IDEPIX_CLOUD_BUFFER)), '
                         'np.logical_not(pixel_classif_flags.IDEPIX_CLOUD_SHADOW)), '
                         'np.logical_not(pixel_classif_flags.IDEPIX_SNOW_ICE)), '
                         'np.logical_not('
                         'np.logical_and(c2rcc_flags.Rtosa_OOS, conc_chl > 1.0))), '
                         'np.logical_not(c2rcc_flags.Rtosa_OOR)), '
                         'np.logical_not(c2rcc_flags.Rhow_OOR)), '
                         'np.logical_not('
                         'np.logical_and(c2rcc_flags.Cloud_risk, immersed_cyanobacteria == 0))),'
                         ' floating_vegetation == 0), conc_chl > 0.01), '
                         'np.logical_not('
                         'np.logical_or(floating_cyanobacteria == 1, chl_pitarch > 500)))',
                         transpile_expr(expr))
