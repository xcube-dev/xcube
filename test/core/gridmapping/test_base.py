import unittest

import numpy as np
import pyproj
import xarray as xr

from test.core.test_geocoding import SourceDatasetMixin
from xcube.core.gridmapping import GridMapping
# noinspection PyProtectedMember
from xcube.core.gridmapping.helpers import _to_affine

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


class TestGridMapping(GridMapping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._xy_coords = xr.DataArray(np.random.random((2, self.height, self.width)),
                                       dims=('coord', 'y', 'x'))

    @property
    def xy_coords(self) -> xr.DataArray:
        return self._xy_coords


# noinspection PyMethodMayBeStatic
class GridMappingTest(SourceDatasetMixin, unittest.TestCase):
    _kwargs = dict(
        size=(7200, 3600),
        tile_size=(3600, 1800),
        xy_bbox=(-180.0, -90.0, 180.0, 90.0),
        xy_res=(360 / 7200, 360 / 7200),
        crs=GEO_CRS,
        is_regular=True,
        is_lon_360=False,
        is_j_axis_up=False,
    )

    def kwargs(self, **kwargs):
        orig_kwargs = dict(self._kwargs)
        orig_kwargs.update(**kwargs)
        return orig_kwargs

    def test_valid(self):
        gm = TestGridMapping(**self.kwargs())
        self.assertEqual((7200, 3600), gm.size)
        self.assertEqual(7200, gm.width)
        self.assertEqual(3600, gm.height)
        self.assertEqual(True, gm.is_tiled)
        self.assertEqual((3600, 1800), gm.tile_size)
        self.assertEqual(3600, gm.tile_width)
        self.assertEqual(1800, gm.tile_height)
        self.assertEqual((0, 0, 7200, 3600), gm.ij_bbox)
        self.assertEqual((-180.0, -90.0, 180.0, 90.0), gm.xy_bbox)
        self.assertEqual(-180.0, gm.x_min)
        self.assertEqual(-90.0, gm.y_min)
        self.assertEqual(180.0, gm.x_max)
        self.assertEqual(90.0, gm.y_max)
        self.assertEqual((0.05, 0.05), gm.xy_res)
        self.assertEqual(0.05, gm.x_res)
        self.assertEqual(0.05, gm.y_res)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(False, gm.is_j_axis_up)

        self.assertIsInstance(gm.xy_coords, xr.DataArray)
        np.testing.assert_equal(np.array([[0, 0, 3599, 1799],
                                          [3600, 0, 7199, 1799],
                                          [0, 1800, 3599, 3599],
                                          [3600, 1800, 7199, 3599]]), gm.ij_bboxes)

    def test_invalids(self):
        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(size=(3600, 1)))
        self.assertEqual('invalid size', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(size=(3600,)))
        self.assertEqual('not enough values to unpack (expected 2, got 1)', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(size=None))
        self.assertEqual('size must be an int or a sequence of two ints', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(tile_size=0))
        self.assertEqual('invalid tile_size', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            TestGridMapping(**self.kwargs(xy_res=-0.1))
        self.assertEqual('invalid xy_res', f'{cm.exception}')

    def test_scalars(self):
        gm = TestGridMapping(**self.kwargs(size=3600, tile_size=1800, xy_res=0.1))
        self.assertEqual((3600, 3600), gm.size)
        self.assertEqual((1800, 1800), gm.tile_size)
        self.assertEqual((0.1, 0.1), gm.xy_res)

    def test_not_tiled(self):
        gm = TestGridMapping(**self.kwargs(tile_size=None))
        self.assertEqual((7200, 3600), gm.tile_size)
        self.assertEqual(False, gm.is_tiled)

    def test_ij_to_xy_transform(self):
        image_geom = TestGridMapping(**self.kwargs(size=(1200, 1200),
                                                   xy_bbox=(0, 0, 1200, 1200),
                                                   xy_res=1,
                                                   crs=NOT_A_GEO_CRS))
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((0, 0), i2crs, (0, 1200))
        self.assertMatrixPoint((1024, 0), i2crs, (1024, 1200))
        self.assertMatrixPoint((0, 1024), i2crs, (0, 1200 - 1024))
        self.assertMatrixPoint((1024, 1024), i2crs, (1024, 1200 - 1024))
        self.assertEqual(((1, 0, 0), (0.0, -1, 1200)), i2crs)

        image_geom = TestGridMapping(**self.kwargs(size=(1440, 720),
                                                   xy_bbox=(-180, -90, 180, 90),
                                                   xy_res=0.25))
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((-180, 90), i2crs, (0, 0))
        self.assertMatrixPoint((0, 0), i2crs, (720, 360))
        self.assertMatrixPoint((180, -90), i2crs, (1440, 720))
        self.assertEqual(((0.25, 0.0, -180.0), (0.0, -0.25, 90.0)), i2crs)

        image_geom = TestGridMapping(**self.kwargs(size=(1440, 720),
                                                   xy_bbox=(-180, -90, 180, 90),
                                                   xy_res=0.25,
                                                   is_j_axis_up=True))
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((-180, -90), i2crs, (0, 0))
        self.assertMatrixPoint((0, 0), i2crs, (720, 360))
        self.assertMatrixPoint((180, 90), i2crs, (1440, 720))
        self.assertEqual(((0.25, 0.0, -180.0), (0.0, 0.25, -90.0)), i2crs)

    def test_xy_to_ij_transform(self):
        image_geom = TestGridMapping(**self.kwargs(size=(1200, 1200),

                                                   xy_bbox=(0, 0, 1200, 1200),
                                                   xy_res=1,
                                                   crs=NOT_A_GEO_CRS))
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 0), crs2i, (0, 1200))
        self.assertMatrixPoint((1024, 0), crs2i, (1024, 1200))
        self.assertMatrixPoint((0, 1024), crs2i, (0, 1200 - 1024))
        self.assertMatrixPoint((1024, 1024), crs2i, (1024, 1200 - 1024))
        self.assertEqual(((1, 0, 0), (0.0, -1, 1200)), crs2i)

        image_geom = TestGridMapping(**self.kwargs(size=(1440, 720),
                                                   xy_bbox=(-180, -90, 180, 90),
                                                   xy_res=0.25))
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 720), crs2i, (-180, -90))
        self.assertMatrixPoint((720, 360), crs2i, (0, 0))
        self.assertMatrixPoint((1440, 0), crs2i, (180, 90))
        self.assertEqual(((4.0, 0.0, 720.0), (0.0, -4.0, 360.0)), crs2i)

        image_geom = TestGridMapping(**self.kwargs(size=(1440, 720),
                                                   xy_bbox=(-180, -90, 180, 90),
                                                   xy_res=0.25,
                                                   is_j_axis_up=True))
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 0), crs2i, (-180, -90))
        self.assertMatrixPoint((720, 360), crs2i, (0, 0))
        self.assertMatrixPoint((1440, 720), crs2i, (180, 90))
        self.assertEqual(((4.0, 0.0, 720.0), (0.0, 4.0, 360.0)), crs2i)

    def test_ij_transform_from(self):
        source = TestGridMapping(**self.kwargs(size=(1440, 720),
                                               xy_bbox=(-180, -90, 180, 90),
                                               xy_res=0.25,
                                               is_j_axis_up=True))
        target = TestGridMapping(**self.kwargs(size=(1000, 1000),
                                               xy_bbox=(10, 50, 10 + 0.025 * 1000, 50 + 0.025 * 1000),
                                               xy_res=0.025,
                                               is_j_axis_up=True))
        combined = source.ij_transform_from(target)

        im2crs = target.ij_to_xy_transform
        crs2im = source.xy_to_ij_transform

        crs_point = self.assertMatrixPoint((10, 50), im2crs, (0, 0))
        im_point = self.assertMatrixPoint((760, 560), crs2im, crs_point)
        self.assertMatrixPoint(im_point, combined, (0, 0))

        crs_point = self.assertMatrixPoint((22.5, 56.25), im2crs, (500, 250))
        im_point = self.assertMatrixPoint((810, 585), crs2im, crs_point)
        self.assertMatrixPoint(im_point, combined, (500, 250))

        self.assertEqual(((0.1, 0.0, 760.0),
                          (0.0, 0.1, 560.0)), combined)

    def assertMatrixPoint(self, expected_point, matrix, point):
        affine = _to_affine(matrix)
        actual_point = affine * point
        self.assertAlmostEqual(expected_point[0], actual_point[0])
        self.assertAlmostEqual(expected_point[1], actual_point[1])
        return actual_point

    def test_derive(self):
        gm = TestGridMapping(**self.kwargs())
        self.assertEqual((7200, 3600), gm.size)
        self.assertEqual((3600, 1800), gm.tile_size)
        self.assertEqual(False, gm.is_j_axis_up)
        derived_gm = gm.derive(tile_size=512, is_j_axis_up=True)
        self.assertIsNot(gm, derived_gm)
        self.assertIsInstance(derived_gm, TestGridMapping)
        self.assertEqual((7200, 3600), derived_gm.size)
        self.assertEqual((512, 512), derived_gm.tile_size)
        self.assertEqual(True, derived_gm.is_j_axis_up)
