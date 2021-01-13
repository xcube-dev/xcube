import unittest

import numpy as np
import xarray as xr

from xcube.core.gridmapping.bboxes import compute_ij_bboxes


class ComputeIJBBoxesTest(unittest.TestCase):

    def setUp(self) -> None:
        lon = xr.DataArray(np.linspace(10., 20., 11), dims='x')
        lat = xr.DataArray(np.linspace(50., 60., 11), dims='y')
        lat, lon = xr.broadcast(lat, lon)
        self.lon_values = lon.values
        self.lat_values = lat.values

    def test_all_included(self):
        self._assert_all_included(compute_ij_bboxes)

    def test_tiles(self):
        self._assert_tiles(compute_ij_bboxes)

    def test_none_found(self):
        self._assert_none_found(compute_ij_bboxes)

    def test_with_border(self):
        self._assert_with_border(compute_ij_bboxes)

    def _assert_all_included(self, compute_ij_bboxes_func):
        xy_bboxes = np.array([[10., 50., 20., 60.]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes_func(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 10, 10]], dtype=np.int64))

    def _assert_tiles(self, compute_ij_bboxes_func):
        a0 = 0.
        a1 = a0 + 5.
        a2 = a1 + 5.
        xy_bboxes = np.array([[10. + a0, 50. + a0, 10. + a1, 50. + a1],
                              [10. + a1, 50. + a0, 10. + a2, 50. + a1],
                              [10. + a0, 50. + a1, 10. + a1, 50. + a2],
                              [10. + a1, 50. + a1, 10. + a2, 50. + a2]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes_func(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 5, 5],
                                                 [5, 0, 10, 5],
                                                 [0, 5, 5, 10],
                                                 [5, 5, 10, 10]], dtype=np.int64))

    def _assert_none_found(self, compute_ij_bboxes_func):
        a0 = 11.
        a1 = a0 + 5.
        a2 = a1 + 5.
        xy_bboxes = np.array([[10. + a0, 50. + a0, 10. + a1, 50. + a1],
                              [10. + a1, 50. + a0, 10. + a2, 50. + a1],
                              [10. + a0, 50. + a1, 10. + a1, 50. + a2],
                              [10. + a1, 50. + a1, 10. + a2, 50. + a2]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes_func(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[-1, -1, -1, -1],
                                                 [-1, -1, -1, -1],
                                                 [-1, -1, -1, -1],
                                                 [-1, -1, -1, -1]], dtype=np.int64))

    def _assert_with_border(self, compute_ij_bboxes_func):
        xy_bboxes = np.array([[12.4, 51.6, 12.6, 51.7]])

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes_func(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[-1, -1, -1, -1]], dtype=np.int64))

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes_func(self.lon_values, self.lat_values, xy_bboxes, 0.5, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[2, 2, 3, 2]], dtype=np.int64))

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes_func(self.lon_values, self.lat_values, xy_bboxes, 1.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[2, 1, 3, 2]], dtype=np.int64))

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes_func(self.lon_values, self.lat_values, xy_bboxes, 2.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[1, 0, 4, 3]], dtype=np.int64))

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes_func(self.lon_values, self.lat_values, xy_bboxes, 2.0, 2, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 6, 5]], dtype=np.int64))
