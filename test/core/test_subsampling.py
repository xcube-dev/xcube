#  The MIT License (MIT)
#  Copyright (c) 2022 by the xcube development team and contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

import unittest

import numpy as np
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.subsampling import find_agg_method
from xcube.core.subsampling import get_dataset_agg_methods
from xcube.core.subsampling import subsample_dataset
from xcube.constants import CRS84


class SubsampleDatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        test_data_1 = np.array([[1, 2, 3, 4, 5, 6],
                                [2, 3, 4, 5, 6, 7],
                                [3, 4, 5, 6, 7, 8],
                                [4, 5, 6, 7, 8, 9],
                                [1, 2, 3, 4, 5, 6]], dtype=np.int16)
        test_data_1 = np.stack([test_data_1, test_data_1 + 10])
        test_data_2 = 0.1 * test_data_1
        self.dataset = new_cube(width=6,  # even
                                height=5,  # odd
                                x_name='x',
                                y_name='y',
                                x_start=0.5,
                                y_start=1.5,
                                x_res=1.,
                                y_res=1.,
                                time_periods=2,
                                crs=CRS84,
                                crs_name='spatial_ref',
                                variables=dict(var_1=test_data_1,
                                               var_2=test_data_2))

    def test_subsample_dataset_none(self):
        subsampled_dataset = subsample_dataset(self.dataset,
                                               step=2,
                                               agg_methods=None)
        self.assert_subsampling_ok(
            subsampled_dataset,
            np.array([
                [[1, 3, 5],
                 [3, 5, 7],
                 [1, 3, 5]],

                [[11, 13, 15],
                 [13, 15, 17],
                 [11, 13, 15]]
            ], dtype=np.int16),
            np.array([
                [[0.2, 0.4, 0.6],
                 [0.4, 0.6, 0.8],
                 [0.15, 0.35, 0.55]],

                [[1.2, 1.4, 1.6],
                 [1.4, 1.6, 1.8],
                 [1.15, 1.35, 1.55]]
            ], dtype=np.float64),
            np.array([1., 3., 5.]),
            np.array([2., 4., 6.]),
        )

    def test_subsample_dataset_first(self):
        subsampled_dataset = subsample_dataset(self.dataset,
                                               step=2,
                                               agg_methods='first')
        self.assert_subsampling_ok(
            subsampled_dataset,
            np.array([
                [[1, 3, 5],
                 [3, 5, 7],
                 [1, 3, 5]],

                [[11, 13, 15],
                 [13, 15, 17],
                 [11, 13, 15]]
            ], dtype=np.int16),
            np.array([
                [[0.1, 0.3, 0.5],
                 [0.3, 0.5, 0.7],
                 [0.1, 0.3, 0.5]],

                [[1.1, 1.3, 1.5],
                 [1.3, 1.5, 1.7],
                 [1.1, 1.3, 1.5]]
            ], dtype=np.float64),
            np.array([1., 3., 5.]),
            np.array([2., 4., 6.]),
        )

    def test_subsample_dataset_mean(self):
        subsampled_dataset = subsample_dataset(self.dataset,
                                               step=2,
                                               agg_methods='mean')
        self.assert_subsampling_ok(
            subsampled_dataset,
            np.array([
                [[2, 4, 6],
                 [4, 6, 8],
                 [1, 3, 5]],

                [[12, 14, 16],
                 [14, 16, 18],
                 [11, 13, 15]]
            ], dtype=np.int16),
            np.array([
                [[0.2, 0.4, 0.6],
                 [0.4, 0.6, 0.8],
                 [0.15, 0.35, 0.55]],

                [[1.2, 1.4, 1.6],
                 [1.4, 1.6, 1.8],
                 [1.15, 1.35, 1.55]]
            ], dtype=np.float64),
            np.array([1., 3., 5.]),
            np.array([2., 4., 6.]),
        )

    def test_subsample_dataset_max(self):
        subsampled_dataset = subsample_dataset(self.dataset,
                                               step=2,
                                               agg_methods='max')
        self.assert_subsampling_ok(
            subsampled_dataset,
            np.array([
                [[3, 5, 7],
                 [5, 7, 9],
                 [2, 4, 6]],

                [[13, 15, 17],
                 [15, 17, 19],
                 [12, 14, 16]]
            ], dtype=np.int16),
            np.array([
                [[0.3, 0.5, 0.7],
                 [0.5, 0.7, 0.9],
                 [0.2, 0.4, 0.6]],

                [[1.3, 1.5, 1.7],
                 [1.5, 1.7, 1.9],
                 [1.2, 1.4, 1.6]]
            ], dtype=np.float64),
            np.array([1., 3., 5.]),
            np.array([2., 4., 6.]),
        )

    def assert_subsampling_ok(self,
                              subsampled_dataset: xr.Dataset,
                              expected_var_1: np.ndarray,
                              expected_var_2: np.ndarray,
                              expected_x: np.ndarray,
                              expected_y: np.ndarray):
        self.assertIsInstance(subsampled_dataset, xr.Dataset)
        self.assertIn('spatial_ref',
                      subsampled_dataset)
        self.assertIn('grid_mapping_name',
                      subsampled_dataset.spatial_ref.attrs)

        self.assertIn('x',
                      subsampled_dataset.coords)
        self.assertIn('y',
                      subsampled_dataset.coords)

        np.testing.assert_array_equal(
            subsampled_dataset.x.values,
            expected_x
        )
        np.testing.assert_array_almost_equal(
            subsampled_dataset.y.values,
            expected_y
        )

        self.assertIn('var_1',
                      subsampled_dataset.data_vars)
        self.assertIn('var_2',
                      subsampled_dataset.data_vars)

        self.assertIn('grid_mapping',
                      subsampled_dataset.var_1.attrs)
        self.assertIn('grid_mapping',
                      subsampled_dataset.var_2.attrs)

        self.assertEqual(expected_var_1.dtype,
                         subsampled_dataset.var_1.dtype)
        self.assertEqual(expected_var_2.dtype,
                         subsampled_dataset.var_2.dtype)

        np.testing.assert_array_equal(
            subsampled_dataset.var_1.values,
            expected_var_1
        )
        np.testing.assert_array_almost_equal(
            subsampled_dataset.var_2.values,
            expected_var_2
        )

    def test_get_dataset_agg_methods(self):
        agg_methods = get_dataset_agg_methods(self.dataset,
                                              agg_methods=None)
        self.assertEqual({'var_1': 'first', 'var_2': 'mean'},
                         agg_methods)

        agg_methods = get_dataset_agg_methods(self.dataset,
                                              agg_methods='mean')
        self.assertEqual({'var_1': 'mean', 'var_2': 'mean'},
                         agg_methods)

        agg_methods = get_dataset_agg_methods(self.dataset,
                                              agg_methods='max')
        self.assertEqual({'var_1': 'max', 'var_2': 'max'},
                         agg_methods)

    # noinspection PyTypeChecker
    def test_find_agg_method(self):
        for m in ('first', 'min', 'max', 'mean', 'median'):
            self.assertEqual(m, find_agg_method(m, 'var_1', np.uint8))
            self.assertEqual(m, find_agg_method(m, 'var_2', np.float32))

        for m in ({'*': 'max'}, {'var_*': 'max'}):
            self.assertEqual('max',
                             find_agg_method(m, 'var_1', np.uint8))
            self.assertEqual('max',
                             find_agg_method(m, 'var_2', np.float32))

        for m in (None, 'auto'):
            self.assertEqual('first', find_agg_method(m, 'var_1', np.uint8))
            self.assertEqual('mean', find_agg_method(m, 'var_2', np.float32))

        for m in ({'*': None}, {'*': 'auto'},
                  {'var_*': None}, {'var_*': 'auto'}):
            self.assertEqual('first', find_agg_method(m, 'var_1', np.uint8))
            self.assertEqual('mean', find_agg_method(m, 'var_2', np.float32))
