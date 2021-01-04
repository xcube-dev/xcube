import unittest
from typing import Tuple

import numpy as np
# noinspection PyUnresolvedReferences
import xarray as xr

from xcube.core.rectify import ImageGeom
from xcube.core.rectify import rectify_dataset
from .test_geocoding import SourceDatasetMixin

nan = np.nan


# noinspection PyMethodMayBeStatic
class RectifyDatasetTest(SourceDatasetMixin, unittest.TestCase):

    def test_rectify_2x2_to_default(self):
        src_ds = self.new_source_dataset()

        dst_ds = rectify_dataset(src_ds)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (4, 4))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0., 2., 4., 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 54., 52., 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, nan, nan, nan],
                                           [nan, 1.0, 2.0, nan],
                                           [3.0, 3.0, 2.0, nan],
                                           [nan, 4.0, nan, nan]
                                       ],
                                           dtype=rad.dtype))

    def test_rectify_2x2_to_7x7(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(7, 7), x_min=-0.5, y_min=49.5, xy_res=1.0)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (7, 7))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5., 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55., 54., 53., 52., 51., 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, 1.0, nan, nan, nan, nan, nan],
                                           [nan, 1.0, 1.0, nan, nan, nan, nan],
                                           [nan, 1.0, 1.0, 1.0, 2.0, nan, nan],
                                           [nan, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0],
                                           [3.0, 3.0, 3.0, 4.0, 2.0, nan, nan],
                                           [nan, 3.0, 4.0, 4.0, nan, nan, nan],
                                           [nan, nan, 4.0, nan, nan, nan, nan],
                                       ], dtype=rad.dtype))

    def test_rectify_2x2_to_7x7_subset(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(7, 7), x_min=1.5, y_min=50.5, xy_res=1.0)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (7, 7))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([2.0, 3.0, 4.0, 5., 6., 7., 8.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([57., 56., 55., 54., 53., 52., 51.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values,
                                       np.array([
                                           [nan, nan, nan, nan, nan, nan, nan],
                                           [nan, nan, nan, nan, nan, nan, nan],
                                           [1.0, nan, nan, nan, nan, nan, nan],
                                           [1.0, 1.0, 2.0, nan, nan, nan, nan],
                                           [3.0, 1.0, 2.0, 2.0, 2.0, nan, nan],
                                           [3.0, 4.0, 2.0, nan, nan, nan, nan],
                                           [4.0, 4.0, nan, nan, nan, nan, nan],
                                       ], dtype=rad.dtype))

    def test_rectify_2x2_to_13x13(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (13, 13))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55.5, 55., 54.5, 54., 53.5, 53., 52.5, 52., 51.5, 51.,
                                                 50.5, 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_j_axis_up(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5, is_j_axis_up=True)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (13, 13))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([50., 50.5, 51., 51.5, 52., 52.5, 53., 53.5, 54., 54.5, 55.,
                                                 55.5, 56.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype)[::-1])

    def test_rectify_2x2_to_13x13_j_axis_up_dask_5x5(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5, tile_size=5, is_j_axis_up=True)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (13, 13), chunks=((5, 5, 3), (5, 5, 3)))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([50., 50.5, 51., 51.5, 52., 52.5, 53., 53.5, 54., 54.5, 55.,
                                                 55.5, 56.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype)[::-1])

    def test_rectify_2x2_to_13x13_dask_7x7(self):
        src_ds = self.new_source_dataset()
        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5, tile_size=7)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (13, 13), chunks=((7, 6), (7, 6)))

        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55.5, 55., 54.5, 54., 53.5, 53., 52.5, 52., 51.5, 51.,
                                                 50.5, 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_5x5(self):
        src_ds = self.new_source_dataset()
        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5, tile_size=5)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (13, 13), chunks=((5, 5, 3), (5, 5, 3)))

        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55.5, 55., 54.5, 54., 53.5, 53., 52.5, 52., 51.5, 51.,
                                                 50.5, 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_3x13(self):
        src_ds = self.new_source_dataset()
        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5, tile_size=(3, 13))
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (13, 13), chunks=((13,), (3, 3, 3, 3, 1)))

        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55.5, 55., 54.5, 54., 53.5, 53., 52.5, 52., 51.5, 51.,
                                                 50.5, 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_dask_13x3(self):
        src_ds = self.new_source_dataset()
        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5, tile_size=(13, 3))
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (13, 13), chunks=((3, 3, 3, 3, 1), (13,)))

        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55.5, 55., 54.5, 54., 53.5, 53., 52.5, 52., 51.5, 51.,
                                                 50.5, 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_antimeridian(self):
        src_ds = self.new_source_dataset_antimeridian()

        output_geom = ImageGeom(size=(13, 13), x_min=177.75, y_min=49.75, xy_res=0.5, is_geo_crs=True)
        self.assertEqual(True, output_geom.is_lon_360)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNotNone(dst_ds)
        lon, lat, rad = self._assert_shape_and_dim(dst_ds, (13, 13))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([178., 178.5, 179., 179.5, 180., -179.5, -179., -178.5,
                                                 -178., -177.5, -177., -176.5, -176.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55.5, 55., 54.5, 54., 53.5, 53., 52.5, 52., 51.5, 51.,
                                                 50.5, 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

    def test_rectify_2x2_to_13x13_output_ij_names(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom, output_ij_names=('source_i', 'source_j'))
        lon, lat, rad, source_i, source_j = self._assert_shape_and_dim(dst_ds, (13, 13),
                                                                       var_names=('rad', 'source_i', 'source_j'))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55.5, 55., 54.5, 54., 53.5, 53., 52.5, 52., 51.5, 51.,
                                                 50.5, 50.],
                                                dtype=lat.dtype))
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))
        np.testing.assert_almost_equal(np.floor(source_i.values + 0.5), self.expected_i_13x13())
        np.testing.assert_almost_equal(np.floor(source_j.values + 0.5), self.expected_j_13x13())

    def test_rectify_2x2_to_13x13_output_ij_names_dask(self):
        src_ds = self.new_source_dataset()

        output_geom = ImageGeom(size=(13, 13), x_min=-0.25, y_min=49.75, xy_res=0.5, tile_size=5)

        dst_ds = rectify_dataset(src_ds, output_geom=output_geom, output_ij_names=('source_i', 'source_j'))
        lon, lat, rad, source_i, source_j = self._assert_shape_and_dim(dst_ds, (13, 13),
                                                                       chunks=((5, 5, 3), (5, 5, 3)),
                                                                       var_names=('rad', 'source_i', 'source_j'))
        np.testing.assert_almost_equal(lon.values,
                                       np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5., 5.5, 6.],
                                                dtype=lon.dtype))
        np.testing.assert_almost_equal(lat.values,
                                       np.array([56., 55.5, 55., 54.5, 54., 53.5, 53., 52.5, 52., 51.5, 51.,
                                                 50.5, 50.],
                                                dtype=lat.dtype))
        print('source_i:', np.floor(source_i.values + 0.5))
        print('source_j:', np.floor(source_j.values + 0.5))
        np.testing.assert_almost_equal(np.floor(source_i.values + 0.5), self.expected_i_13x13())
        np.testing.assert_almost_equal(np.floor(source_j.values + 0.5), self.expected_j_13x13())
        np.testing.assert_almost_equal(rad.values, self.expected_rad_13x13(rad.dtype))

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

    def _assert_shape_and_dim(self, dst_ds, size, chunks=None, var_names=('rad',)) -> Tuple[xr.DataArray, ...]:
        w, h = size

        self.assertIn('lon', dst_ds)
        lon = dst_ds['lon']
        self.assertEqual((w,), lon.shape)
        self.assertEqual(('lon',), lon.dims)

        self.assertIn('lat', dst_ds)
        lat = dst_ds['lat']
        self.assertEqual((h,), lat.shape)
        self.assertEqual(('lat',), lat.dims)

        # noinspection PyShadowingBuiltins
        vars = []
        for var_name in var_names:
            self.assertIn(var_name, dst_ds)
            var = dst_ds[var_name]
            self.assertEqual((h, w), var.shape)
            self.assertEqual(('lat', 'lon'), var.dims)
            self.assertEqual(chunks, var.chunks)
            vars.append(var)

        return (lon, lat, *vars)

    def test_compute_and_extract_source_pixels(self):
        from xcube.core.rectify import _compute_ij_images_numpy_parallel
        from xcube.core.rectify import _compute_var_image_numpy_parallel
        self._assert_compute_and_extract_source_pixels(_compute_ij_images_numpy_parallel,
                                                       _compute_var_image_numpy_parallel, False)
        from xcube.core.rectify import _compute_ij_images_numpy_sequential
        from xcube.core.rectify import _compute_var_image_numpy_sequential
        self._assert_compute_and_extract_source_pixels(_compute_ij_images_numpy_sequential,
                                                       _compute_var_image_numpy_sequential, False)

    def test_compute_and_extract_source_pixels_j_axis_up(self):
        from xcube.core.rectify import _compute_ij_images_numpy_parallel
        from xcube.core.rectify import _compute_var_image_numpy_parallel
        self._assert_compute_and_extract_source_pixels(_compute_ij_images_numpy_parallel,
                                                       _compute_var_image_numpy_parallel, True)
        from xcube.core.rectify import _compute_ij_images_numpy_sequential
        from xcube.core.rectify import _compute_var_image_numpy_sequential
        self._assert_compute_and_extract_source_pixels(_compute_ij_images_numpy_sequential,
                                                       _compute_var_image_numpy_sequential, True)

    def _assert_compute_and_extract_source_pixels(self,
                                                  compute_ij_images,
                                                  compute_var_image,
                                                  is_j_axis_up: bool):
        src_ds = self.new_source_dataset()

        dst_src_ij = np.full((2, 13, 13), np.nan, dtype=np.float64)
        dst_x_offset = -0.25
        dst_y_offset = 49.75 if is_j_axis_up else 56.25
        dst_x_scale = 0.5
        dst_y_scale = 0.5 if is_j_axis_up else -0.5
        compute_ij_images(src_ds.lon.values,
                          src_ds.lat.values,
                          0,
                          0,
                          dst_src_ij,
                          dst_x_offset,
                          dst_y_offset,
                          dst_x_scale,
                          dst_y_scale,
                          1e-5)

        if not is_j_axis_up:
            np.testing.assert_almost_equal(np.floor(dst_src_ij[0] + 0.5), self.expected_i_13x13())
            np.testing.assert_almost_equal(np.floor(dst_src_ij[1] + 0.5), self.expected_j_13x13())
        else:
            np.testing.assert_almost_equal(np.floor(dst_src_ij[0] + 0.5), self.expected_i_13x13()[::-1])
            np.testing.assert_almost_equal(np.floor(dst_src_ij[1] + 0.5), self.expected_j_13x13()[::-1])

        dst_rad = np.full((13, 13), np.nan, dtype=np.float64)

        compute_var_image(src_ds.rad.values,
                          dst_src_ij,
                          dst_rad)

        if not is_j_axis_up:
            np.testing.assert_almost_equal(dst_rad, self.expected_rad_13x13(dst_rad.dtype))
        else:
            np.testing.assert_almost_equal(dst_rad, self.expected_rad_13x13(dst_rad.dtype)[::-1])

    def expected_i_13x13(self):
        return np.array([
            [nan, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, nan, nan, nan, nan, nan],
            [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, nan, nan, nan, nan],
            [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, nan, nan],
            [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan],
            [nan, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan],
            [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan],
        ],
            dtype=np.float64)

    def expected_j_13x13(self):
        return np.array([
            [nan, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan],
            [nan, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan],
            [nan, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, nan],
            [nan, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, nan, nan],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, nan, nan, nan],
            [nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan],
            [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan],
        ],
            dtype=np.float64)

    def expected_rad_13x13(self, dtype):
        return np.array([
            [nan, nan, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, nan, nan, nan, nan],
            [nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, nan, nan, nan, nan],
            [nan, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
            [nan, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [nan, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0, 2.0, nan, nan],
            [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, nan, nan, nan],
            [nan, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, nan, nan, nan, nan],
            [nan, nan, 3.0, 4.0, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, 4.0, 4.0, 4.0, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, 4.0, nan, nan, nan, nan, nan, nan, nan, nan],
        ], dtype=dtype)


class RectifySentinel2DatasetTest(SourceDatasetMixin, unittest.TestCase):

    def test_rectify_dataset(self):
        src_ds = create_s2plus_dataset()

        expected_data = np.array([
            [nan, nan, nan, nan, 0.009, 0.009, 0.012, 0.012, 0.012, 0.023, nan],
            [nan, 0.028, 0.021, 0.021, 0.01, 0.01, 0.01, 0.009, 0.009, 0.023, nan],
            [nan, 0.028, 0.021, 0.021, 0.01, 0.01, 0.01, 0.009, 0.009, 0.023, nan],
            [nan, 0.037, 0.023, 0.023, 0.008, 0.008, 0.008, 0.01, 0.009, 0.023, nan],
            [nan, 0.041, 0.023, 0.023, 0.007, 0.007, 0.007, 0.01, 0.01, 0.021, nan],
            [nan, 0.041, 0.023, 0.023, 0.007, 0.007, 0.007, 0.01, 0.01, 0.021, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        ])
        dst_ds = rectify_dataset(src_ds, tile_size=None)
        print(dst_ds.rrs_665.values)
        self.assertEqual(None, dst_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(dst_ds.rrs_665.values, expected_data, decimal=3)

        dst_ds = rectify_dataset(src_ds, tile_size=5)
        self.assertEqual(((5, 2), (5, 5, 1)), dst_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(dst_ds.rrs_665.values, expected_data, decimal=3)

        dst_ds = rectify_dataset(src_ds, tile_size=None, is_j_axis_up=True)
        self.assertEqual(None, dst_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(dst_ds.rrs_665.values, expected_data[::-1], decimal=3)

        dst_ds = rectify_dataset(src_ds, tile_size=5, is_j_axis_up=True)
        self.assertEqual(((5, 2), (5, 5, 1)), dst_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(dst_ds.rrs_665.values, expected_data[::-1], decimal=3)


def create_s2plus_dataset():
    x = xr.DataArray([310005., 310015., 310025., 310035., 310045.], dims=["x"],
                     attrs=dict(units="m", standard_name="projection_x_coordinate"))
    y = xr.DataArray([5689995., 5689985., 5689975., 5689965., 5689955.], dims=["y"],
                     attrs=dict(units="m", standard_name="projection_y_coordinate"))
    lon = xr.DataArray([[0.272763, 0.272906, 0.273050, 0.273193, 0.273336],
                        [0.272768, 0.272911, 0.273055, 0.273198, 0.273342],
                        [0.272773, 0.272917, 0.273060, 0.273204, 0.273347],
                        [0.272779, 0.272922, 0.273066, 0.273209, 0.273352],
                        [0.272784, 0.272927, 0.273071, 0.273214, 0.273358]],
                       dims=["y", "x"], attrs=dict(units="degrees_east", standard_name="longitude"))
    lat = xr.DataArray([[51.329464, 51.329464, 51.329468, 51.32947, 51.329475],
                        [51.329372, 51.329376, 51.32938, 51.329384, 51.329388],
                        [51.329285, 51.329285, 51.32929, 51.329292, 51.329296],
                        [51.329193, 51.329197, 51.32920, 51.329205, 51.329205],
                        [51.329100, 51.329105, 51.32911, 51.329113, 51.329117]],
                       dims=["y", "x"], attrs=dict(units="degrees_north", standard_name="latitude"))
    rrs_443 = xr.DataArray([[0.014000, 0.014000, 0.016998, 0.016998, 0.016998],
                            [0.014000, 0.014000, 0.016998, 0.016998, 0.016998],
                            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998],
                            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998],
                            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998]],
                           dims=["y", "x"], attrs=dict(units="sr-1", grid_mapping="transverse_mercator"))
    rrs_665 = xr.DataArray([[0.025002, 0.019001, 0.008999, 0.012001, 0.022999],
                            [0.028000, 0.021000, 0.009998, 0.008999, 0.022999],
                            [0.036999, 0.022999, 0.007999, 0.008999, 0.023998],
                            [0.041000, 0.022999, 0.007000, 0.009998, 0.021000],
                            [0.033001, 0.018002, 0.007999, 0.008999, 0.021000]],
                           dims=["y", "x"], attrs=dict(units="sr-1", grid_mapping="transverse_mercator"))
    transverse_mercator = xr.DataArray(np.array([0xffffffff], dtype=np.uint32),
                                       attrs=dict(grid_mapping_name="transverse_mercator",
                                                  scale_factor_at_central_meridian=0.9996,
                                                  longitude_of_central_meridian=3.0,
                                                  latitude_of_projection_origin=0.0,
                                                  false_easting=500000.0,
                                                  false_northing=0.0,
                                                  semi_major_axis=6378137.0,
                                                  inverse_flattening=298.257223563))
    return xr.Dataset(dict(rrs_443=rrs_443, rrs_665=rrs_665, transverse_mercator=transverse_mercator),
                      coords=dict(x=x, y=y, lon=lon, lat=lat),
                      attrs={
                          "title": "T31UCS_20180802T105621",
                          "conventions": "CF-1.6",
                          "institution": "VITO",
                          "product_type": "DCS4COP Sentinel2 Product",
                          "origin": "Copernicus Sentinel Data",
                          "project": "DCS4COP",
                          "time_coverage_start": "2018-08-02T10:59:38.888000Z",
                          "time_coverage_end": "2018-08-02T10:59:38.888000Z"
                      })
