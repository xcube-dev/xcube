import unittest

import numpy as np

from xcube.core.rectify import ImageGeom
from xcube.core.rectify import compute_source_pixels
from xcube.core.rectify import extract_source_pixels
from xcube.core.rectify import rectify_dataset
from .test_geocoding import SourceDatasetMixin

nan = np.nan


class RectifyDatasetTest(SourceDatasetMixin, unittest.TestCase):

    def test_rectify_2x2_to_default(self):
        src_ds = self.new_source_dataset()

        dst_ds = rectify_dataset(src_ds)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, 4, 4)
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 2.0, 4.0, 6.0],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([50.0, 52.0, 54.0, 56.0],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, 4.0, nan, nan],
                                           [3.0, 3.0, 2.0, nan],
                                           [nan, 1.0, 2.0, nan],
                                           [nan, nan, nan, nan]
                                       ], dtype=rad.dtype))

    def test_rectify_2x2_to_7x7(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(7, 7), x_min=0.0, y_min=50.0, xy_res=1.0)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, 7, 7)
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, nan, 4.0, nan, nan, nan, nan],
                                           [nan, 3.0, 4.0, 4.0, nan, nan, nan],
                                           [3.0, 3.0, 3.0, 4.0, 2.0, nan, nan],
                                           [nan, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0],
                                           [nan, 1.0, 1.0, 1.0, 2.0, nan, nan],
                                           [nan, 1.0, 1.0, nan, nan, nan, nan],
                                           [nan, 1.0, nan, nan, nan, nan, nan],
                                       ], dtype=rad.dtype))

    def test_rectify_2x2_to_7x7_subset(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(7, 7), x_min=2.0, y_min=51.0, xy_res=1.0)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, 7, 7)
        np.testing.assert_almost_equal(lon.values,
                                       np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [4.0, 4.0, nan, nan, nan, nan, nan],
                                           [3.0, 4.0, 2.0, nan, nan, nan, nan],
                                           [3.0, 1.0, 2.0, 2.0, 2.0, nan, nan],
                                           [1.0, 1.0, 2.0, nan, nan, nan, nan],
                                           [1.0, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, nan, nan, nan, nan, nan],
                                       ], dtype=rad.dtype))

    def test_rectify_2x2_to_13x13(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=0.0, y_min=50.0, xy_res=0.5)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, 13, 13)
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.5,
                                                 56.0],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, nan, nan, nan, 4.0, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, nan, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 3.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, nan, nan, nan, nan],
                                           [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                                           [nan, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                       ], dtype=rad.dtype))

    def test_rectify_2x2_to_13x13_inverse_y_axis(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=0.0, y_min=50.0, xy_res=0.5)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom, is_y_axis_inverted=True)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, 13, 13)
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.5,
                                                 56.0],
                                                dtype=lat.dtype)[::-1])
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, nan, nan, nan, 4.0, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, nan, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 3.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, nan, nan, nan, nan],
                                           [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                                           [nan, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                       ], dtype=rad.dtype)[::-1])

    def test_rectify_2x2_to_13x13_dask(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=0.0, y_min=50.0, xy_res=0.5, tile_size=7)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, 13, 13)
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.5,
                                                 56.0],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, nan, nan, nan, 4.0, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, nan, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 3.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, nan, nan, nan, nan],
                                           [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                                           [nan, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                       ], dtype=rad.dtype))

    def test_rectify_2x2_to_13x13_antimeridian(self):
        src_ds = self.new_source_dataset_antimeridian()

        output_geom = ImageGeom(size=(13, 13), x_min=+178.0, y_min=+50.0, xy_res=0.5)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNotNone(dst_ds)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, 13, 13)
        np.testing.assert_almost_equal(lon.values,
                                       np.array([+178.0, +178.5, +179.0, +179.5, +180.0,
                                                 -179.5, -179.0, -178.5, -178.0, -177.5, -177.0, -176.5, -176.0],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([50.0, 50.5, 51.0, 51.5, 52.0,
                                                 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.5, 56.0],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, nan, nan, nan, 4.0, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, nan, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 3.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, nan, nan, nan, nan],
                                           [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.],
                                           [nan, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                       ], dtype=rad.dtype))

    def test_rectify_2x2_to_13x13_none(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=10.0, y_min=50.0, xy_res=0.5)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNone(dst_ds)

        output_geom = ImageGeom(size=(13, 13), x_min=-10.0, y_min=50.0, xy_res=0.5)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNone(dst_ds)

        output_geom = ImageGeom(size=(13, 13), x_min=0.0, y_min=58.0, xy_res=0.5)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNone(dst_ds)

        output_geom = ImageGeom(size=(13, 13), x_min=0.0, y_min=42.0, xy_res=0.5)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNone(dst_ds)

    def _assert_shape_and_dim(self, dst_ds, w, h):
        self.assertIn('lon', dst_ds)
        lon = dst_ds['lon']
        self.assertEqual((w,), lon.shape)
        self.assertEqual(('lon',), lon.dims)

        self.assertIn('lat', dst_ds)
        lat = dst_ds['lat']
        self.assertEqual((h,), lat.shape)
        self.assertEqual(('lat',), lat.dims)

        self.assertIn('rad', dst_ds)
        rad = dst_ds['rad']
        self.assertEqual((h, w), rad.shape)
        self.assertEqual(('lat', 'lon'), rad.dims)

        return lon, lat, rad

    def test_compute_source_pixels(self):
        src_ds = self.new_source_dataset()

        dst_src_i = np.full((13, 13), np.nan, dtype=np.float64)
        dst_src_j = np.full((13, 13), np.nan, dtype=np.float64)
        compute_source_pixels(src_ds.lon.values,
                              src_ds.lat.values,
                              0,
                              0,
                              dst_src_i,
                              dst_src_j,
                              0.0,
                              50.0,
                              0.5)

        # print(xr.DataArray(src_i, dims=('y', 'x')))
        # print(xr.DataArray(dst_src_j, dims=('y', 'x')))

        np.testing.assert_almost_equal(np.floor(dst_src_i + 0.5),
                                       np.array([[nan, nan, nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan],
                                                 [nan, nan, nan, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan],
                                                 [nan, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan],
                                                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan],
                                                 [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan],
                                                 [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                 [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, nan, nan],
                                                 [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, nan, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]],
                                                dtype=np.float64))
        np.testing.assert_almost_equal(np.floor(dst_src_j + 0.5),
                                       np.array([[nan, nan, nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan],
                                                 [nan, nan, nan, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                                                 [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan],
                                                 [nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, nan, nan, nan, nan],
                                                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, nan, nan, nan],
                                                 [nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, nan, nan],
                                                 [nan, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 [nan, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan],
                                                 [nan, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                                 [nan, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]],
                                                dtype=np.float64))

        dst_rad = np.zeros((13, 13), dtype=np.float64)

        extract_source_pixels(src_ds.rad.values,
                              dst_src_i,
                              dst_src_j,
                              dst_rad)

        np.testing.assert_almost_equal(dst_rad,
                                       np.array([
                                           [nan, nan, nan, nan, 4.0, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, nan, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 3.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, nan, nan, nan, nan],
                                           [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, nan, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                                           [nan, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
                                       ],
                                           dtype=np.float64))
