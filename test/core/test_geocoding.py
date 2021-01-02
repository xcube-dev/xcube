import unittest

import numpy as np
import pyproj as pp
import xarray as xr

from xcube.core.geocoding import CRS_WGS84
from xcube.core.geocoding import GeoCoding
from xcube.core.geocoding import compute_ij_bboxes
from xcube.core.geocoding import gu_compute_ij_bboxes


def _new_source_dataset():
    lon = np.array([[1.0, 6.0],
                    [0.0, 2.0]])
    lat = np.array([[56.0, 53.0],
                    [52.0, 50.0]])
    rad = np.array([[1.0, 2.0],
                    [3.0, 4.0]])
    return xr.Dataset(dict(lon=xr.DataArray(lon, dims=('y', 'x')),
                           lat=xr.DataArray(lat, dims=('y', 'x')),
                           rad=xr.DataArray(rad, dims=('y', 'x'))))


def _new_source_dataset_antimeridian():
    lon = np.array([[+179.0, -176.0],
                    [+178.0, +180.0]])
    lat = np.array([[56.0, 53.0],
                    [52.0, 50.0]])
    rad = np.array([[1.0, 2.0],
                    [3.0, 4.0]])
    return xr.Dataset(dict(lon=xr.DataArray(lon, dims=('y', 'x')),
                           lat=xr.DataArray(lat, dims=('y', 'x')),
                           rad=xr.DataArray(rad, dims=('y', 'x'))))


class SourceDatasetMixin:

    @classmethod
    def new_source_dataset(cls):
        return _new_source_dataset()

    @classmethod
    def new_source_dataset_antimeridian(cls):
        return _new_source_dataset_antimeridian()


class GeoCodingTest(SourceDatasetMixin, unittest.TestCase):
    def test_is_geo_crs_and_is_lon_normalized(self):
        x = xr.DataArray(np.linspace(10.0, 20.0, 21), dims='columns', name='lon')
        y = xr.DataArray(np.linspace(53.0, 58.0, 11), dims='rows', name='lat')
        y, x = xr.broadcast(y, x)

        gc = GeoCoding(x, y)
        self.assertEqual(CRS_WGS84, gc.crs)
        self.assertEqual(True, gc.is_geo_crs)
        self.assertEqual(False, gc.is_lon_normalized)

        gc = GeoCoding(x, y, is_geo_crs=True)
        self.assertEqual(CRS_WGS84, gc.crs)
        self.assertEqual(True, gc.is_geo_crs)
        self.assertEqual(False, gc.is_lon_normalized)

        gc = GeoCoding(x, y, is_lon_normalized=True)
        self.assertEqual(CRS_WGS84, gc.crs)
        self.assertEqual(True, gc.is_geo_crs)
        self.assertEqual(True, gc.is_lon_normalized)

        gc = GeoCoding(x, y, is_lon_normalized=False)
        self.assertEqual(CRS_WGS84, gc.crs)
        self.assertEqual(True, gc.is_geo_crs)
        self.assertEqual(False, gc.is_lon_normalized)

        with self.assertRaises(ValueError) as cm:
            gc = GeoCoding(x, y, crs=pp.crs.CRS(32633), is_geo_crs=True)
        self.assertEqual('crs and is_geo_crs are inconsistent',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            gc = GeoCoding(x, y, is_geo_crs=False, is_lon_normalized=True)
        self.assertEqual('is_geo_crs and is_lon_normalized are inconsistent',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            gc = GeoCoding(x, y, crs=pp.crs.CRS(32633), is_lon_normalized=True)
        self.assertEqual('crs and is_lon_normalized are inconsistent',
                         f'{cm.exception}')

    def test_from_dataset_1d_rectified(self):
        x = xr.DataArray(np.linspace(10.0, 20.0, 21), dims='x')
        y = xr.DataArray(np.linspace(53.0, 58.0, 11), dims='y')
        gc = GeoCoding.from_dataset(xr.Dataset(dict(x=x, y=y), attrs=CRS_WGS84.to_cf()))
        self.assertIsInstance(gc.x, xr.DataArray)
        self.assertIsInstance(gc.y, xr.DataArray)
        self.assertEqual('x', gc.x_name)
        self.assertEqual('y', gc.y_name)
        self.assertEqual(True, gc.is_rectified)
        self.assertEqual(CRS_WGS84, gc.crs)
        self.assertEqual(True, gc.is_geo_crs)
        self.assertEqual(False, gc.is_lon_normalized)

    def test_from_dataset_2d_rectified(self):
        x = xr.DataArray(np.linspace(10.0, 20.0, 21), dims='columns')
        y = xr.DataArray(np.linspace(53.0, 58.0, 11), dims='rows')
        y, x = xr.broadcast(y, x)
        gc = GeoCoding.from_dataset(xr.Dataset(dict(x=x, y=y), attrs=CRS_WGS84.to_cf()))
        self.assertIsInstance(gc.x, xr.DataArray)
        self.assertIsInstance(gc.y, xr.DataArray)
        self.assertEqual('x', gc.x_name)
        self.assertEqual('y', gc.y_name)
        self.assertEqual(True, gc.is_rectified)
        self.assertEqual(CRS_WGS84, gc.crs)
        self.assertEqual(True, gc.is_geo_crs)
        self.assertEqual(False, gc.is_lon_normalized)

    def test_from_dataset_2d_not_rectified(self):
        x = xr.DataArray(np.linspace(10.0, 20.0, 21), dims='columns')
        y = xr.DataArray(np.linspace(53.0, 58.0, 11), dims='rows')
        y, x = xr.broadcast(y, x)
        # Add noise to x, y so they are no longer rectified
        x = x + 0.01 * np.random.random_sample((11, 21))
        y = y + 0.01 * np.random.random_sample((11, 21))
        gc = GeoCoding.from_dataset(xr.Dataset(dict(x=x, y=y), attrs=CRS_WGS84.to_cf()))
        self.assertIsInstance(gc.x, xr.DataArray)
        self.assertIsInstance(gc.y, xr.DataArray)
        self.assertEqual('x', gc.x_name)
        self.assertEqual('y', gc.y_name)
        self.assertEqual(False, gc.is_rectified)
        self.assertEqual(CRS_WGS84, gc.crs)
        self.assertEqual(True, gc.is_geo_crs)
        self.assertEqual(False, gc.is_lon_normalized)

    def test_ij_bbox(self):
        self._test_ij_bbox(conservative=False)

    def test_ij_bbox_antimeridian(self):
        self._test_ij_bbox_antimeridian(conservative=False)

    def test_ij_bbox_conservative(self):
        self._test_ij_bbox(conservative=True)

    def test_ij_bbox_conservative_antimeridian(self):
        self._test_ij_bbox_antimeridian(conservative=True)

    def _test_ij_bbox(self, conservative: bool):
        x = xr.DataArray(np.linspace(10.0, 20.0, 21), dims='x')
        y = xr.DataArray(np.linspace(53.0, 58.0, 11), dims='y')
        y, x = xr.broadcast(y, x)
        gc = GeoCoding(x=x, y=y, x_name='x', y_name='y', is_geo_crs=True)
        ij_bbox = gc.ij_bbox_conservative if conservative else gc.ij_bbox
        self.assertEqual((-1, -1, -1, -1), ij_bbox((0, -50, 30, 0)))
        self.assertEqual((0, 0, 20, 10), ij_bbox((0, 50, 30, 60)))
        self.assertEqual((0, 0, 20, 6), ij_bbox((0, 50, 30, 56)))
        self.assertEqual((10, 0, 20, 6), ij_bbox((15, 50, 30, 56)))
        self.assertEqual((10, 0, 16, 6), ij_bbox((15, 50, 18, 56)))
        self.assertEqual((10, 1, 16, 6), ij_bbox((15, 53.5, 18, 56)))
        self.assertEqual((8, 0, 18, 8), ij_bbox((15, 53.5, 18, 56), ij_border=2))

    def _test_ij_bbox_antimeridian(self, conservative: bool):
        def denorm(x):
            return x if x <= 180 else x - 360

        lon = xr.DataArray(np.linspace(175.0, 185.0, 21), dims='columns')
        lat = xr.DataArray(np.linspace(53.0, 58.0, 11), dims='rows')
        lat, lon = xr.broadcast(lat, lon)
        gc = GeoCoding(x=lon, y=lat, x_name='lon', y_name='lat', is_lon_normalized=True)
        ij_bbox = gc.ij_bbox_conservative if conservative else gc.ij_bbox
        self.assertEqual((-1, -1, -1, -1), ij_bbox((0, -50, 30, 0)))
        self.assertEqual((0, 0, 20, 10), ij_bbox((denorm(160), 50, denorm(200), 60)))
        self.assertEqual((0, 0, 20, 6), ij_bbox((denorm(160), 50, denorm(200), 56)))
        self.assertEqual((10, 0, 20, 6), ij_bbox((denorm(180), 50, denorm(200), 56)))
        self.assertEqual((10, 0, 16, 6), ij_bbox((denorm(180), 50, denorm(183), 56)))
        self.assertEqual((10, 1, 16, 6), ij_bbox((denorm(180), 53.5, denorm(183), 56)))
        self.assertEqual((8, 0, 18, 8), ij_bbox((denorm(180), 53.5, denorm(183), 56), ij_border=2))
        self.assertEqual((12, 1, 20, 6), ij_bbox((denorm(181), 53.5, denorm(200), 56)))
        self.assertEqual((12, 1, 18, 6), ij_bbox((denorm(181), 53.5, denorm(184), 56)))

    def test_ij_bboxes(self):
        x = xr.DataArray(np.linspace(10.0, 20.0, 21), dims='x')
        y = xr.DataArray(np.linspace(53.0, 58.0, 11), dims='y')
        y, x = xr.broadcast(y, x)
        gc = GeoCoding(x=x, y=y, x_name='x', y_name='y', is_geo_crs=True)

        ij_bboxes = gc.ij_bboxes(np.array([(0.0, -50.0, 30.0, 0.0)]))
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([(-1, -1, -1, -1)], dtype=np.int64))

        ij_bboxes = gc.ij_bboxes(np.array([(0.0, 50, 30, 60),
                                           (0.0, 50, 30, 56),
                                           (15, 50, 30, 56),
                                           (15, 50, 18, 56),
                                           (15, 53.5, 18, 56)]))
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([(0, 0, 20, 10),
                                                 (0, 0, 20, 6),
                                                 (10, 0, 20, 6),
                                                 (10, 0, 16, 6),
                                                 (10, 1, 16, 6)], dtype=np.int64))

    def test_ij_bboxes_gu(self):
        x = xr.DataArray(np.linspace(10.0, 20.0, 21), dims='x')
        y = xr.DataArray(np.linspace(53.0, 58.0, 11), dims='y')
        y, x = xr.broadcast(y, x)
        gc = GeoCoding(x=x, y=y, x_name='x', y_name='y', is_geo_crs=True)

        ij_bboxes = gc.ij_bboxes(np.array([(0.0, -50.0, 30.0, 0.0)]), gu=True)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([(-1, -1, -1, -1)], dtype=np.int64))

        ij_bboxes = gc.ij_bboxes(np.array([(0.0, 50, 30, 60),
                                           (0.0, 50, 30, 56),
                                           (15, 50, 30, 56),
                                           (15, 50, 18, 56),
                                           (15, 53.5, 18, 56)]), gu=True)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([(0, 0, 20, 10),
                                                 (0, 0, 20, 6),
                                                 (10, 0, 20, 6),
                                                 (10, 0, 16, 6),
                                                 (10, 1, 16, 6)], dtype=np.int64))


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

    def test_all_included_gu(self):
        self._assert_all_included(gu_compute_ij_bboxes)

    def test_tiles_gu(self):
        self._assert_tiles(gu_compute_ij_bboxes)

    def test_none_found_gu(self):
        self._assert_none_found(gu_compute_ij_bboxes)

    def test_with_border_gu(self):
        self._assert_with_border(gu_compute_ij_bboxes)

    def _assert_all_included(self, compute_ij_bboxes):
        xy_bboxes = np.array([[10., 50., 20., 60.]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 10, 10]], dtype=np.int64))

    def _assert_tiles(self, compute_ij_bboxes):
        a0 = 0.
        a1 = a0 + 5.
        a2 = a1 + 5.
        xy_bboxes = np.array([[10. + a0, 50. + a0, 10. + a1, 50. + a1],
                              [10. + a1, 50. + a0, 10. + a2, 50. + a1],
                              [10. + a0, 50. + a1, 10. + a1, 50. + a2],
                              [10. + a1, 50. + a1, 10. + a2, 50. + a2]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 5, 5],
                                                 [5, 0, 10, 5],
                                                 [0, 5, 5, 10],
                                                 [5, 5, 10, 10]], dtype=np.int64))

    def _assert_none_found(self, compute_ij_bboxes):
        a0 = 11.
        a1 = a0 + 5.
        a2 = a1 + 5.
        xy_bboxes = np.array([[10. + a0, 50. + a0, 10. + a1, 50. + a1],
                              [10. + a1, 50. + a0, 10. + a2, 50. + a1],
                              [10. + a0, 50. + a1, 10. + a1, 50. + a2],
                              [10. + a1, 50. + a1, 10. + a2, 50. + a2]])
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[-1, -1, -1, -1],
                                                 [-1, -1, -1, -1],
                                                 [-1, -1, -1, -1],
                                                 [-1, -1, -1, -1]], dtype=np.int64))

    def _assert_with_border(self, compute_ij_bboxes):
        xy_bboxes = np.array([[12.4, 51.6, 12.6, 51.7]])

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[-1, -1, -1, -1]], dtype=np.int64))

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 0.5, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[2, 2, 3, 2]], dtype=np.int64))

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 1.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[2, 1, 3, 2]], dtype=np.int64))

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 2.0, 0, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[1, 0, 4, 3]], dtype=np.int64))

        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        compute_ij_bboxes(self.lon_values, self.lat_values, xy_bboxes, 2.0, 2, ij_bboxes)
        np.testing.assert_almost_equal(ij_bboxes,
                                       np.array([[0, 0, 6, 5]], dtype=np.int64))
