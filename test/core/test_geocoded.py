import unittest

import numpy as np
import xarray as xr

from xcube.core.geocoded import compute_output_geom, reproject_dataset, ImageGeom

TEST_INPUT = '../xcube-gen-bc/test/inputdata/O_L2_0001_SNS_2017104102450_v1.0.nc'

nan = np.nan


class ReprojectTest(unittest.TestCase):

    def new_source_dataset(self):
        lon = np.array([[1.0, 6.0],
                        [0.0, 2.0]])
        lat = np.array([[56.0, 53.0],
                        [52.0, 50.0]])
        rad = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
        return xr.Dataset(dict(lon=xr.DataArray(lon, dims=('y', 'x')),
                               lat=xr.DataArray(lat, dims=('y', 'x')),
                               rad=xr.DataArray(rad, dims=('y', 'x'))))

    def new_source_dataset_antimeridian(self):
        lon = np.array([[+179.0, -176.0],
                        [+178.0, +180.0]])
        lat = np.array([[56.0, 53.0],
                        [52.0, 50.0]])
        rad = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
        return xr.Dataset(dict(lon=xr.DataArray(lon, dims=('y', 'x')),
                               lat=xr.DataArray(lat, dims=('y', 'x')),
                               rad=xr.DataArray(rad, dims=('y', 'x'))))

    def test_reproject_2x2_to_default(self):
        src_ds = self.new_source_dataset()

        dst_ds = reproject_dataset(src_ds)
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

    def test_reproject_2x2_to_7x7(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(width=7, height=7, x_min=0.0, y_min=50.0, res=1.0)

        dst_ds = reproject_dataset(src_ds, output_geom=output_geom)
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

    def test_reproject_2x2_to_13x13(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(width=13, height=13, x_min=0.0, y_min=50.0, res=0.5)

        dst_ds = reproject_dataset(src_ds, output_geom=output_geom)
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
                                           [nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.],
                                           [nan, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
                                           [nan, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                                       ], dtype=rad.dtype))

    def test_reproject_2x2_to_13x13_antimeridian(self):
        src_ds = self.new_source_dataset_antimeridian()

        output_geom = ImageGeom(width=13, height=13, x_min=+178.0, y_min=+50.0, res=0.5)

        dst_ds = reproject_dataset(src_ds, output_geom=output_geom)
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

    def test_compute_output_geom(self):
        src_ds = self.new_source_dataset()

        self._assert_image_geom(ImageGeom(4, 4, 0.0, 50.0, 2.0),
                                compute_output_geom(src_ds))

        self._assert_image_geom(ImageGeom(7, 7, 0.0, 50.0, 1.0),
                                compute_output_geom(src_ds,
                                                    oversampling=2.0))

        self._assert_image_geom(ImageGeom(8, 8, 0.0, 50.0, 1.0),
                                compute_output_geom(src_ds,
                                                    denom_x=4,
                                                    denom_y=4,
                                                    oversampling=2.0))

    def test_compute_output_geom_antimeridian(self):
        src_ds = self.new_source_dataset_antimeridian()

        self._assert_image_geom(ImageGeom(4, 4, 178.0, 50.0, 2.0),
                                compute_output_geom(src_ds))

        self._assert_image_geom(ImageGeom(7, 7, 178.0, 50.0, 1.0),
                                compute_output_geom(src_ds,
                                                    oversampling=2.0))

        self._assert_image_geom(ImageGeom(8, 8, 178.0, 50.0, 1.0),
                                compute_output_geom(src_ds,
                                                    denom_x=4,
                                                    denom_y=4,
                                                    oversampling=2.0))

    def _assert_image_geom(self,
                           expected: ImageGeom,
                           actual: ImageGeom):
        self.assertEqual(expected.width, actual.width)
        self.assertEqual(expected.height, actual.height)
        self.assertAlmostEqual(actual.x_min, actual.x_min, delta=1e-5)
        self.assertAlmostEqual(actual.y_min, actual.y_min, delta=1e-5)
        self.assertAlmostEqual(actual.res, actual.res, delta=1e-6)

    def test_image_geom_is_crossing_antimeridian(self):
        output_geom = ImageGeom(width=13, height=13, x_min=0.0, y_min=+50.0, res=0.5)
        self.assertFalse(output_geom.is_crossing_antimeridian)

        output_geom = ImageGeom(width=13, height=13, x_min=178.0, y_min=+50.0, res=0.5)
        self.assertTrue(output_geom.is_crossing_antimeridian)
