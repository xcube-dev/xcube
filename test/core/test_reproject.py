import unittest

import numpy as np
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt
from numpy.testing import assert_array_almost_equal

from test.sampledata import create_highroc_dataset
from xcube.core.reproject import reproject_xy_to_wgs84

nan = np.nan


class ReprojectTest(unittest.TestCase):

    def test_reproject_xy_to_wgs84_highroc(self):
        dst_width = 12
        dst_height = 9

        dataset = create_highroc_dataset()
        proj_dataset = reproject_xy_to_wgs84(dataset,
                                             src_xy_var_names=('lon', 'lat'),
                                             src_xy_tp_var_names=('TP_longitude', 'TP_latitude'),
                                             src_xy_gcp_step=1,
                                             src_xy_tp_gcp_step=1,
                                             dst_size=(dst_width, dst_height))

        self.assertIsNotNone(proj_dataset)
        self.assertEqual(dict(lon=dst_width, lat=dst_height, bnds=2),
                         proj_dataset.sizes)

        self.assertIn('lon', proj_dataset)
        self.assertEqual(proj_dataset.lon.shape, (dst_width,))
        self.assertIn('lat', proj_dataset)
        self.assertEqual(proj_dataset.lat.shape, (dst_height,))

        self.assertIn('lon_bnds', proj_dataset)
        self.assertEqual(proj_dataset.lon_bnds.shape, (dst_width, 2))
        self.assertIn('lat_bnds', proj_dataset)
        self.assertEqual(proj_dataset.lat_bnds.shape, (dst_height, 2))

        expected_conc_chl = np.array([[7., 7., 11., 11., 11., 11., nan, nan, nan, nan, 5., 5.],
                                      [7., 7., 11., 11., 11., 11., nan, nan, nan, 21., 21., 21.],
                                      [5., 5., 10., 10., 10., 10., 2., 2., 2., 21., 21., 21.],
                                      [5., 5., 10., 10., 10., 2., 2., 2., 2., 21., 17., 17.],
                                      [5., 5., 10., 10., 10., 20., 20., 20., 20., 17., 17., 17.],
                                      [5., 16., 6., 6., 6., 20., 20., 20., 17., 17., nan, nan],
                                      [16., 16., 6., 6., 6., 20., nan, nan, nan, nan, nan, nan],
                                      [16., 16., 6., nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                      [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]],
                                     dtype=np.float64)
        self.assertIn('conc_chl', proj_dataset)
        # print(proj_dataset.conc_chl)
        self.assertEqual(proj_dataset.conc_chl.shape, (dst_height, dst_width))
        self.assertEqual(proj_dataset.conc_chl.dtype, np.float64)
        assert_array_almost_equal(proj_dataset.conc_chl, expected_conc_chl)

        expected_c2rcc_flags = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2.],
                                         [1., 1., 4., 4., 4., 4., 1., 1., 1., 2., 2., 2.],
                                         [1., 1., 4., 4., 4., 1., 1., 1., 1., 2., 1., 1.],
                                         [1., 1., 4., 4., 4., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 8., 1., 1., 1., 1., 1., 1., 1., 1., nan, nan],
                                         [8., 8., 1., 1., 1., 1., nan, nan, nan, nan, nan, nan],
                                         [8., 8., 1., nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                         [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]],
                                        dtype=np.float64)
        self.assertIn('c2rcc_flags', proj_dataset)
        # print(proj_dataset.c2rcc_flags)
        self.assertEqual(proj_dataset.c2rcc_flags.shape, (dst_height, dst_width))
        self.assertEqual(proj_dataset.c2rcc_flags.dtype, np.float64)
        assert_array_almost_equal(proj_dataset.c2rcc_flags, expected_c2rcc_flags)
