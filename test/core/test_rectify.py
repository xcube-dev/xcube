import unittest
from typing import Tuple

import numpy as np
# noinspection PyUnresolvedReferences
import xarray as xr

from test.sampledata import SourceDatasetMixin
from test.sampledata import create_s2plus_dataset
from xcube.core.gridmapping import CRS_WGS84
from xcube.core.gridmapping import GridMapping
from xcube.core.rectify import rectify_dataset

nan = np.nan


# noinspection PyMethodMayBeStatic
class RectifyDatasetTest(SourceDatasetMixin, unittest.TestCase):

    def test_rectify_2x2_to_default(self):
        src_ds = self.new_source_dataset()

        output_geom = GridMapping.regular(size=(4, 4),
                                          xy_min=(-1, 49),
                                          xy_res=2,
                                          crs=CRS_WGS84)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        # dst_ds = rectify_dataset(src_ds)

        rad = dst_ds.rad
        # lon, lat, rad = self._assert_shape_and_dim(dst_ds, (4, 4))
        # np.testing.assert_almost_equal(lon.values,
        #                                np.array([0., 2., 4., 6.],
        #                                         dtype=lon.dtype))
        # np.testing.assert_almost_equal(lat.values,
        #                                np.array([56., 54., 52., 50.],
        #                                         dtype=lat.dtype))
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

        output_geom = GridMapping.regular(size=(7, 7),
                                          xy_min=(-0.5, 49.5),
                                          xy_res=1.0,
                                          crs=CRS_WGS84)

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

        output_geom = GridMapping.regular(size=(7, 7),
                                          xy_min=(1.5, 50.5),
                                          xy_res=1.0,
                                          crs=CRS_WGS84)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84,
                                          is_j_axis_up=True)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84,
                                          tile_size=5,
                                          is_j_axis_up=True)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84,
                                          tile_size=7)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84,
                                          tile_size=5)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84,
                                          tile_size=(3, 13))

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84,
                                          tile_size=(13, 3))

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(177.75, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-0.25, 49.75),
                                          xy_res=0.5,
                                          crs=CRS_WGS84,
                                          tile_size=5)

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

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(10.0, 50.0),
                                          xy_res=0.5,
                                          crs=CRS_WGS84)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNone(dst_ds)

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(-10.0, 50.0),
                                          xy_res=0.5,
                                          crs=CRS_WGS84)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNone(dst_ds)

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(0.0, 58.0),
                                          xy_res=0.5,
                                          crs=CRS_WGS84)
        dst_ds = rectify_dataset(src_ds, output_geom=output_geom)
        self.assertIsNone(dst_ds)

        output_geom = GridMapping.regular(size=(13, 13),
                                          xy_min=(0.0, 42.0),
                                          xy_res=0.5,
                                          crs=CRS_WGS84)
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
            [nan, nan, nan, nan, nan, nan, nan],
            [nan, 0.019001, 0.008999, 0.008999, 0.012001, 0.022999, nan],
            [nan, 0.022999, 0.022999, 0.007999, 0.008999, 0.023998, nan],
            [nan, 0.022999, 0.022999, 0.007, 0.009998, 0.021, nan],
            [nan, nan, nan, nan, nan, nan, nan]
        ])

        source_gm = GridMapping.from_dataset(src_ds, prefer_crs=CRS_WGS84)

        dst_ds = rectify_dataset(src_ds, geo_coding=source_gm, tile_size=None)
        print(dst_ds.rrs_665.values)
        self.assertEqual(None, dst_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(dst_ds.rrs_665.values, expected_data, decimal=3)

        dst_ds = rectify_dataset(src_ds, geo_coding=source_gm, tile_size=5)
        self.assertEqual(((5,), (5, 2)), dst_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(dst_ds.rrs_665.values, expected_data, decimal=3)

        dst_ds = rectify_dataset(src_ds, geo_coding=source_gm, tile_size=None, is_j_axis_up=True)
        self.assertEqual(None, dst_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(dst_ds.rrs_665.values, expected_data[::-1], decimal=3)

        dst_ds = rectify_dataset(src_ds, geo_coding=source_gm, tile_size=5, is_j_axis_up=True)
        self.assertEqual(((5,), (5, 2)), dst_ds.rrs_665.chunks)
        np.testing.assert_almost_equal(dst_ds.rrs_665.values, expected_data[::-1], decimal=3)
