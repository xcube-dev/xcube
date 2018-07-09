import unittest

import numpy as np
import xarray as xr

from test.sampledata import create_highroc_dataset
from xcube.dsutil import select_variables, update_variable_props, compute_dataset, add_time_coords, \
    get_time_in_days_since_1970

nan = float('nan')


class ComputeDatasetTest(unittest.TestCase):
    @classmethod
    def get_test_dataset(cls):
        return xr.Dataset(dict(a=(('y', 'x'), [[0.1, 0.2, 0.4, 0.1], [0.5, 0.1, 0.2, 0.3]]),
                               b=(('y', 'x'), [[0.4, 0.3, 0.2, 0.4], [0.1, 0.2, 0.5, 0.1]],
                                  dict(valid_pixel_expression='a >= 0.2')),
                               c=((), 1.0),
                               d=((), 1.0, dict(expression='a * b'))),
                          coords=dict(x=(('x',), [1, 2, 3, 4]), y=(('y',), [1, 2])),
                          attrs=dict(title='test_compute_dataset'))

    def test_compute_dataset_without_processed_variables(self):
        dataset = self.get_test_dataset()
        computed_dataset = compute_dataset(dataset)
        self.assertIsNot(computed_dataset, dataset)
        self.assertIn('x', computed_dataset)
        self.assertIn('y', computed_dataset)
        self.assertIn('a', computed_dataset)
        self.assertIn('b', computed_dataset)
        self.assertIn('c', computed_dataset)
        self.assertIn('d', computed_dataset)
        self.assertIn('x', computed_dataset.coords)
        self.assertIn('y', computed_dataset.coords)
        self.assertIn('title', computed_dataset.attrs)
        self.assertEqual((2, 4), computed_dataset.a.shape)
        self.assertEqual((2, 4), computed_dataset.b.shape)
        self.assertEqual((), computed_dataset.c.shape)
        self.assertNotIn('expression', computed_dataset.c.attrs)
        self.assertEqual((2, 4), computed_dataset.d.shape)
        self.assertIn('expression', computed_dataset.d.attrs)
        np.testing.assert_array_almost_equal(computed_dataset.a.values,
                                             np.array([[0.1, 0.2, 0.4, 0.1], [0.5, 0.1, 0.2, 0.3]]))
        np.testing.assert_array_almost_equal(computed_dataset.b.values,
                                             np.array([[nan, 0.3, 0.2, nan], [0.1, nan, 0.5, 0.1]]))
        np.testing.assert_array_almost_equal(computed_dataset.c.values,
                                             np.array([1.]))
        np.testing.assert_array_almost_equal(computed_dataset.d.values,
                                             np.array([[0.04, 0.06, 0.08, 0.04],
                                                       [0.05, 0.02, 0.1, 0.03]]))

    def test_compute_dataset_with_processed_variables(self):
        dataset = self.get_test_dataset()
        computed_dataset = compute_dataset(dataset,
                                           processed_variables=[('a', None),
                                                                ('b', None),
                                                                ('c', dict(expression='a + b')),
                                                                ('d', dict(valid_pixel_expression='c > 0.4'))])
        self.assertIsNot(computed_dataset, dataset)
        self.assertIn('x', computed_dataset)
        self.assertIn('y', computed_dataset)
        self.assertIn('a', computed_dataset)
        self.assertIn('b', computed_dataset)
        self.assertIn('c', computed_dataset)
        self.assertIn('d', computed_dataset)
        self.assertIn('x', computed_dataset.coords)
        self.assertIn('y', computed_dataset.coords)
        self.assertIn('title', computed_dataset.attrs)
        self.assertEqual((2, 4), computed_dataset.a.shape)
        self.assertEqual((2, 4), computed_dataset.b.shape)
        self.assertEqual((2, 4), computed_dataset.c.shape)
        self.assertIn('expression', computed_dataset.c.attrs)
        self.assertEqual((2, 4), computed_dataset.d.shape)
        self.assertIn('expression', computed_dataset.d.attrs)
        np.testing.assert_array_almost_equal(computed_dataset.a.values,
                                             np.array([[0.1, 0.2, 0.4, 0.1], [0.5, 0.1, 0.2, 0.3]]))
        np.testing.assert_array_almost_equal(computed_dataset.b.values,
                                             np.array([[nan, 0.3, 0.2, nan], [0.1, nan, 0.5, 0.1]]))
        np.testing.assert_array_almost_equal(computed_dataset.c.values,
                                             np.array([[nan, 0.5, 0.6, nan], [0.6, nan, 0.7, 0.4]]))
        np.testing.assert_array_almost_equal(computed_dataset.d.values,
                                             np.array([[nan, 0.06, 0.08, nan], [0.05, nan, 0.1, nan]]))


class SelectVariablesTest(unittest.TestCase):
    def test_select_all(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = select_variables(ds1, None)
        self.assertIs(ds2, ds1)

    def test_select_none(self):
        ds1 = create_highroc_dataset()
        ds2 = select_variables(ds1, [])
        self.assertEqual(0, len(ds2.data_vars))
        ds2 = select_variables(ds1, ['bibo'])
        self.assertEqual(0, len(ds2.data_vars))

    def test_select_variables_for_some(self):
        ds1 = create_highroc_dataset()
        self.assertEqual(36, len(ds1.data_vars))
        ds2 = select_variables(ds1, ['conc_chl', 'c2rcc_flags', 'rtoa_10'])
        self.assertEqual(3, len(ds2.data_vars))


class UpdateVariablePropsTest(unittest.TestCase):
    def test_no_change(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = update_variable_props(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = update_variable_props(ds1, [])
        self.assertIs(ds2, ds1)

    def test_change_all_or_none(self):
        ds1 = create_highroc_dataset()
        ds2 = update_variable_props(ds1,
                                    [(var_name, {'marker': True}) for var_name in ds1.data_vars])
        self.assertEqual(len(ds1.data_vars), len(ds2.data_vars))
        self.assertTrue(all(['marker' in ds2[n].attrs for n in ds2.variables]))

        with self.assertRaises(KeyError):
            update_variable_props(ds1, [('bibo', {'marker': True})])

    def test_change_some(self):
        ds1 = create_highroc_dataset()
        ds2 = update_variable_props(ds1,
                                    [('conc_chl', {'name': 'chl_c2rcc'}),
                                     ('c2rcc_flags', {'name': 'flags', 'marker': True}),
                                     ('rtoa_10', None)])

        self.assertEqual(len(ds1.data_vars), len(ds2.data_vars))

        self.assertNotIn('conc_chl', ds2.data_vars)
        self.assertNotIn('c2rcc_flags', ds2.data_vars)

        self.assertIn('chl_c2rcc', ds2.data_vars)
        self.assertIn('original_name', ds2.chl_c2rcc.attrs)
        self.assertEqual('conc_chl', ds2.chl_c2rcc.attrs['original_name'])

        self.assertIn('flags', ds2.data_vars)
        self.assertIn('original_name', ds2.flags.attrs)
        self.assertEqual('c2rcc_flags', ds2.flags.attrs['original_name'])
        self.assertIn('marker', ds2.flags.attrs)
        self.assertEqual(True, ds2.flags.attrs['marker'])

        self.assertIn('rtoa_10', ds2.data_vars)

        with self.assertRaises(ValueError) as cm:
            update_variable_props(ds1, [('conc_chl', None),
                                        ('c2rcc_flags', None),
                                        ('rtoa_1', {'name': 'refl_toa'}),
                                        ('rtoa_2', {'name': 'refl_toa'}),
                                        ('rtoa_3', {'name': 'refl_toa'})])
        self.assertEqual("variable 'rtoa_2' cannot be renamed into 'refl_toa' because the name is already in use",
                         f'{cm.exception}')


class AddTimeCoordsTest(unittest.TestCase):

    def test_add_time_coords_point(self):
        dataset = create_highroc_dataset()
        dataset_with_time = add_time_coords(dataset, (365 * 47 + 20, 365 * 47 + 20))
        self.assertIsNot(dataset_with_time, dataset)
        self.assertIn('time', dataset_with_time)
        self.assertEqual(dataset_with_time.time.shape, (1,))
        self.assertNotIn('time_bnds', dataset_with_time)

    def test_add_time_coords_range(self):
        dataset = create_highroc_dataset()
        dataset_with_time = add_time_coords(dataset, (365 * 47 + 20, 365 * 47 + 21))
        self.assertIsNot(dataset_with_time, dataset)
        self.assertIn('time', dataset_with_time)
        self.assertEqual(dataset_with_time.time.shape, (1,))
        self.assertIn('time_bnds', dataset_with_time)
        self.assertEqual(dataset_with_time.time_bnds.shape, (1, 2))

    def test_get_time_in_days_since_1970(self):
        self.assertEqual(17324.5, get_time_in_days_since_1970('201706071200'))
        self.assertEqual(17325.5, get_time_in_days_since_1970('201706081200'))
        self.assertEqual(17690.5, get_time_in_days_since_1970('2018-06-08 12:00'))
        self.assertEqual(17690.5, get_time_in_days_since_1970('2018-06-08T12:00'))
