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
from xcube.core.subsampling import get_dataset_subsampling_slices
from xcube.core.subsampling import subsample_dataset


class SubsampleDatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        test_data = np.array([[1, 2, 3, 4, 5, 6],
                              [2, 3, 4, 5, 6, 7],
                              [3, 4, 5, 6, 7, 8],
                              [4, 5, 6, 7, 8, 9]])
        test_data = np.stack([test_data, test_data + 10])
        self.dataset = new_cube(width=6,
                                height=4,
                                x_name='x',
                                y_name='y',
                                time_periods=2,
                                crs='CRS84',
                                crs_name='spatial_ref',
                                variables=dict(test_var=test_data))

    def test_get_dataset_subsampling_slices(self):
        slices = get_dataset_subsampling_slices(self.dataset,
                                                step=2)
        self.assertEqual(
            {'test_var': (slice(None, None, None),
                          slice(None, None, 2),
                          slice(None, None, 2))
             },
            slices)

    def test_subsample_dataset(self):
        subsampled_dataset = subsample_dataset(self.dataset,
                                               step=2)
        self.assertIsInstance(subsampled_dataset, xr.Dataset)
        self.assertIn('spatial_ref',
                      subsampled_dataset)
        self.assertIn('grid_mapping_name',
                      subsampled_dataset.spatial_ref.attrs)
        self.assertIn('test_var',
                      subsampled_dataset)
        self.assertIn('grid_mapping',
                      subsampled_dataset.test_var.attrs)
        np.testing.assert_array_equal(
            subsampled_dataset.test_var.values,
            np.array([
                [[1, 3, 5],
                 [3, 5, 7]],

                [[11, 13, 15],
                 [13, 15, 17]]
            ])
        )
