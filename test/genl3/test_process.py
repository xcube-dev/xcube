import unittest

import numpy as np
import pandas as pd

from test.sampledata import new_test_dataset
from xcube.genl3.process import resample


class GenerateL3CubeTest(unittest.TestCase):

    def test_resample(self):
        num_times = 30

        time, temperature, precipitation = zip(*[(('2017-07-0%s' if i < 9 else '2017-07-%s') % (i + 1),
                                                  272 + 0.1 * i,
                                                  120 - 0.2 * i) for i in range(num_times)])

        ds1 = new_test_dataset(time, temperature=temperature, precipitation=precipitation)

        ds2 = resample(ds1, '3D', 'max')
        self.assertIsNot(ds2, ds1)
        self.assertIn('time', ds2)
        self.assertIn('temperature', ds2)
        self.assertIn('precipitation', ds2)
        self.assertEqual(('time',), ds2.time.dims)
        self.assertEqual(('time', 'lat', 'lon'), ds2.temperature.dims)
        self.assertEqual(('time', 'lat', 'lon'), ds2.precipitation.dims)
        self.assertEqual((num_times / 3,), ds2.time.shape)
        self.assertEqual((num_times / 3, 180, 360), ds2.temperature.shape)
        self.assertEqual((num_times / 3, 180, 360), ds2.precipitation.shape)
        np.testing.assert_equal(ds2.time.values,
                                np.array(pd.to_datetime(
                                    ['2017-07-01', '2017-07-04', '2017-07-07', '2017-07-10',
                                     '2017-07-13', '2017-07-16', '2017-07-19', '2017-07-22',
                                     '2017-07-25', '2017-07-28'])))
        np.testing.assert_allclose(ds2.temperature.values[..., 0, 0],
                                   np.array([272.2, 272.5, 272.8, 273.1, 273.4, 273.7, 274., 274.3, 274.6, 274.9]))
        np.testing.assert_allclose(ds2.precipitation.values[..., 0, 0],
                                   np.array([120., 119.4, 118.8, 118.2, 117.6, 117., 116.4, 115.8, 115.2, 114.6]))
