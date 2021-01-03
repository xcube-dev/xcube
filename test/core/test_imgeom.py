import unittest

import numpy as np
import pyproj as pp

import xcube.core.new
from xcube.core.geocoding import CRS_WGS84
from xcube.core.rectify import ImageGeom
from xcube.core.sentinel3 import is_sentinel3_product
from xcube.core.sentinel3 import open_sentinel3_product
from .test_geocoding import SourceDatasetMixin

olci_path = 'C :\\Users\\Norman\\Downloads\\S3B_OL_1_EFR____20190728T103451_20190728T103751_20190729T141105_0179_028_108_1800_LN1_O_NT_002.SEN3'


class ImageGeomTest(SourceDatasetMixin, unittest.TestCase):
    def test_invalids(self):
        crs = pp.crs.CRS(4326)

        with self.assertRaises(ValueError) as cm:
            ImageGeom((3600, 0), crs=crs)
        self.assertEqual('invalid size', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom((-3600, 1800), crs=crs)
        self.assertEqual('invalid size', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(1000, tile_size=0, crs=crs)
        self.assertEqual('invalid tile_size', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(1000, tile_size=(100, -100), crs=crs)
        self.assertEqual('invalid tile_size', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(100, xy_res=0.0, crs=crs)
        print(f'{cm.exception}')
        self.assertEqual('invalid xy_res', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(100, xy_res=-0.1, crs=crs)
        print(f'{cm.exception}')
        self.assertEqual('invalid xy_res', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(100, x_min=-190.0, xy_res=0.1, is_geo_crs=True)
        print(f'{cm.exception}')
        self.assertEqual('invalid x_min', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(100, x_min=182.0, xy_res=0.1, is_geo_crs=True)
        self.assertEqual('invalid x_min', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(100, x_min=20.0, xy_res=2.0, is_geo_crs=True)
        self.assertEqual('invalid y_min', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(100, y_min=-100.0, xy_res=0.1, is_geo_crs=True)
        self.assertEqual('invalid y_min', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(100, y_min=100.0, xy_res=0.1, is_geo_crs=True)
        self.assertEqual('invalid y_min', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            ImageGeom(100, y_min=50.0, xy_res=0.5, is_geo_crs=True)
        self.assertEqual('invalid y_min', f'{cm.exception}')

    def test_crs(self):
        image_geom = ImageGeom((2000, 1000))
        self.assertEqual(CRS_WGS84, image_geom.crs)
        self.assertEqual(True, image_geom.is_geo_crs)

    def test_is_j_axis_up(self):
        image_geom = ImageGeom((2000, 1000))
        self.assertEqual(False, image_geom.is_j_axis_up)
        image_geom = ImageGeom((2000, 1000), is_j_axis_up=True)
        self.assertEqual(True, image_geom.is_j_axis_up)

    def test_size(self):
        image_geom = ImageGeom((2000, 1000))
        self.assertEqual((2000, 1000), image_geom.size)

        image_geom = ImageGeom(3600)
        self.assertEqual((3600, 3600), image_geom.size)

        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            ImageGeom(None)

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            ImageGeom((3600, 1800, 4))

    def test_tile_size(self):
        image_geom = ImageGeom(size=(2000, 1000))
        self.assertEqual((2000, 1000), image_geom.size)
        self.assertEqual((2000, 1000), image_geom.tile_size)
        self.assertEqual(False, image_geom.is_tiled)

        image_geom = ImageGeom(size=(2000, 1000), tile_size=None)
        self.assertEqual((2000, 1000), image_geom.size)
        self.assertEqual((2000, 1000), image_geom.tile_size)
        self.assertEqual(False, image_geom.is_tiled)

        image_geom = ImageGeom(size=(2000, 1000), tile_size=(512, 256))
        self.assertEqual((2000, 1000), image_geom.size)
        self.assertEqual((512, 256), image_geom.tile_size)
        self.assertEqual(True, image_geom.is_tiled)

        image_geom = ImageGeom(size=(2000, 1000), tile_size=270)
        self.assertEqual((2000, 1000), image_geom.size)
        self.assertEqual((270, 270), image_geom.tile_size)
        self.assertEqual(True, image_geom.is_tiled)

        image_geom = ImageGeom(size=(400, 200), tile_size=(512, 256))
        self.assertEqual((400, 200), image_geom.size)
        self.assertEqual((400, 200), image_geom.tile_size)
        self.assertEqual(False, image_geom.is_tiled)

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            ImageGeom((2000, 1000), ((512,)))

    def test_is_crossing_antimeridian(self):
        image_geom = ImageGeom(size=100, x_min=0.0, y_min=+50.0, xy_res=0.1, is_geo_crs=True)
        self.assertFalse(image_geom.is_crossing_antimeridian)

        image_geom = ImageGeom(size=100, x_min=178.0, y_min=+50.0, xy_res=0.1, is_geo_crs=True)
        self.assertTrue(image_geom.is_crossing_antimeridian)

    def test_ij_to_xy_transform(self):
        image_geom = ImageGeom(size=(1440, 720),
                               x_min=-180, y_min=-90, xy_res=0.25, is_geo_crs=True)
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((-180, 90), i2crs, (0, 0))
        self.assertMatrixPoint((0, 0), i2crs, (720, 360))
        self.assertMatrixPoint((180, -90), i2crs, (1440, 720))
        self.assertEqual(((0.25, 0.0, -180.0), (0.0, -0.25, 90.0)), i2crs)

        image_geom = ImageGeom(size=(1440, 720), is_j_axis_up=True,
                               x_min=-180, y_min=-90, xy_res=0.25, is_geo_crs=True)
        i2crs = image_geom.ij_to_xy_transform
        self.assertMatrixPoint((-180, -90), i2crs, (0, 0))
        self.assertMatrixPoint((0, 0), i2crs, (720, 360))
        self.assertMatrixPoint((180, 90), i2crs, (1440, 720))
        self.assertEqual(((0.25, 0.0, -180.0), (0.0, 0.25, -90.0)), i2crs)

    def test_xy_to_ij_transform(self):
        image_geom = ImageGeom(size=(1440, 720),
                               x_min=-180, y_min=-90, xy_res=0.25, is_geo_crs=True)
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 720), crs2i, (-180, -90))
        self.assertMatrixPoint((720, 360), crs2i, (0, 0))
        self.assertMatrixPoint((1440, 0), crs2i, (180, 90))
        self.assertEqual(((4.0, 0.0, 720.0), (0.0, -4.0, 360.0)), crs2i)

        image_geom = ImageGeom(size=(1440, 720), is_j_axis_up=True,
                               x_min=-180, y_min=-90, xy_res=0.25, is_geo_crs=True)
        crs2i = image_geom.xy_to_ij_transform
        self.assertMatrixPoint((0, 0), crs2i, (-180, -90))
        self.assertMatrixPoint((720, 360), crs2i, (0, 0))
        self.assertMatrixPoint((1440, 720), crs2i, (180, 90))
        self.assertEqual(((4.0, 0.0, 720.0), (0.0, 4.0, 360.0)), crs2i)

    def test_ij_transform_from(self):
        source = ImageGeom(size=(1440, 720), x_min=-180, y_min=-90, xy_res=0.25, is_j_axis_up=True, is_geo_crs=True)
        target = ImageGeom(size=(1000, 1000), x_min=10, y_min=50, xy_res=0.025, is_j_axis_up=True, is_geo_crs=True)
        combined = source.ij_transform_from(target)

        im2crs = target.ij_to_xy_transform
        crs2im = source.xy_to_ij_transform

        crs_point = self.assertMatrixPoint((10, 50), im2crs, (0, 0))
        im_point = self.assertMatrixPoint((760, 560), crs2im, crs_point)
        self.assertMatrixPoint(im_point, combined, (0, 0))

        crs_point = self.assertMatrixPoint((22.5, 56.25), im2crs, (500, 250))
        im_point = self.assertMatrixPoint((810, 585), crs2im, crs_point)
        self.assertMatrixPoint(im_point, combined, (500, 250))

        self.assertEqual(((0.1, 0.0, 760.0), (0.0, 0.1, 560.0)), combined)

    def assertMatrixPoint(self, expected_point, matrix, point):
        affine = ImageGeom._to_affine(matrix)
        actual_point = affine * point
        self.assertAlmostEqual(expected_point[0], actual_point[0])
        self.assertAlmostEqual(expected_point[1], actual_point[1])
        return actual_point

    def test_derive(self):
        image_geom = ImageGeom((2048, 1024), crs=pp.crs.CRS(32632))
        self.assertEqual((2048, 1024), image_geom.tile_size)
        new_image_geom = image_geom.derive(tile_size=512)
        self.assertIsNot(new_image_geom, image_geom)
        self.assertEqual((2048, 1024), new_image_geom.size)
        self.assertEqual((512, 512), new_image_geom.tile_size)

    def test_xy_bbox(self):
        output_geom = ImageGeom(size=(20, 10), x_min=0.0, y_min=+50.0, xy_res=0.5)
        self.assertEqual((0.0, 50.0, 10.0, 55.0), output_geom.xy_bbox)

    def test_xy_bbox_antimeridian(self):
        output_geom = ImageGeom(size=(20, 10), x_min=174.0, y_min=-30.0, xy_res=0.5, is_geo_crs=True)
        self.assertEqual((174.0, -30.0, -176.0, -25.0), output_geom.xy_bbox)

    def test_ij_bboxes(self):
        image_geom = ImageGeom(size=(2000, 1000), x_min=10.0, y_min=20.0, xy_res=0.1)
        np.testing.assert_almost_equal(image_geom.ij_bboxes,
                                       np.array([[0, 0, 1999, 999]], dtype=np.int64))

        image_geom = ImageGeom(size=(2000, 1000), x_min=10.0, y_min=20.0, xy_res=0.1, tile_size=500)
        np.testing.assert_almost_equal(image_geom.ij_bboxes,
                                       np.array([
                                           [0, 0, 499, 499],
                                           [500, 0, 999, 499],
                                           [1000, 0, 1499, 499],
                                           [1500, 0, 1999, 499],
                                           [0, 500, 499, 999],
                                           [500, 500, 999, 999],
                                           [1000, 500, 1499, 999],
                                           [1500, 500, 1999, 999]
                                       ], dtype=np.int64))

    def test_xy_bboxes(self):
        image_geom = ImageGeom(size=(2000, 1000), x_min=10.0, y_min=20.0, xy_res=0.1)
        np.testing.assert_almost_equal(image_geom.xy_bboxes,
                                       np.array([[10., 20., 209.9, 119.9]], dtype=np.float64))

        image_geom = ImageGeom(size=(2000, 1000), x_min=10.0, y_min=20.0, xy_res=0.1, tile_size=500)
        np.testing.assert_almost_equal(image_geom.xy_bboxes,
                                       np.array([
                                           [10., 20., 59.9, 69.9],
                                           [60., 20., 109.9, 69.9],
                                           [110., 20., 159.9, 69.9],
                                           [160., 20., 209.9, 69.9],
                                           [10., 70., 59.9, 119.9],
                                           [60., 70., 109.9, 119.9],
                                           [110., 70., 159.9, 119.9],
                                           [160., 70., 209.9, 119.9]
                                       ], dtype=np.float64))

    def test_coord_vars(self):
        image_geom = ImageGeom(size=(10, 6), x_min=-2600.0, y_min=1200.0, xy_res=10.0)

        cv = image_geom.coord_vars(xy_names=('x', 'y'))
        self._assert_coord_vars(cv,
                                (10, 6),
                                ('x', 'y'),
                                (-2595., -2505.),
                                (1205., 1255.),
                                ('x_bnds', 'y_bnds'),
                                (
                                    (-2600., -2590.),
                                    (-2510., -2500.),
                                ),
                                (
                                    (1200., 1210.),
                                    (1250., 1260.),
                                ))

    def test_coord_vars_y_reversed(self):
        image_geom = ImageGeom(size=(10, 6), x_min=-2600.0, y_min=1200.0, xy_res=10.0)

        cv = image_geom.coord_vars(xy_names=('x', 'y'), is_y_reversed=True)
        self._assert_coord_vars(cv,
                                (10, 6),
                                ('x', 'y'),
                                (-2595., -2505.),
                                (1255., 1205.),
                                ('x_bnds', 'y_bnds'),
                                (
                                    (-2600., -2590.),
                                    (-2510., -2500.),
                                ),
                                (
                                    (1260., 1250.),
                                    (1210., 1200.),
                                ))

    def test_coord_vars_lon_normalized(self):
        image_geom = ImageGeom(size=(10, 10), x_min=172.0, y_min=53.0, xy_res=2.0, is_geo_crs=True)

        cv = image_geom.coord_vars(xy_names=('lon', 'lat'), is_lon_normalized=True)
        self._assert_coord_vars(cv,
                                (10, 10),
                                ('lon', 'lat'),
                                (173.0, -169.0),
                                (54.0, 72.0),
                                ('lon_bnds', 'lat_bnds'),
                                (
                                    (172., 174.),
                                    (-170., -168.),
                                ),
                                (
                                    (53., 55.),
                                    (71., 73.),
                                ))

    def _assert_coord_vars(self,
                           cv,
                           size,
                           xy_names,
                           x_values,
                           y_values,
                           xy_bnds_names,
                           x_bnds_values,
                           y_bnds_values):
        self.assertIsNotNone(cv)
        self.assertIn(xy_names[0], cv)
        self.assertIn(xy_names[1], cv)
        self.assertIn(xy_bnds_names[0], cv)
        self.assertIn(xy_bnds_names[1], cv)

        x = cv[xy_names[0]]
        self.assertEqual((size[0],), x.shape)
        np.testing.assert_almost_equal(x.values[0], np.array(x_values[0]))
        np.testing.assert_almost_equal(x.values[-1], np.array(x_values[-1]))

        y = cv[xy_names[1]]
        self.assertEqual((size[1],), y.shape)
        np.testing.assert_almost_equal(y.values[0], np.array(y_values[0]))
        np.testing.assert_almost_equal(y.values[-1], np.array(y_values[-1]))

        x_bnds = cv[xy_bnds_names[0]]
        self.assertEqual((size[0], 2), x_bnds.shape)
        np.testing.assert_almost_equal(x_bnds.values[0], np.array(x_bnds_values[0]))
        np.testing.assert_almost_equal(x_bnds.values[-1], np.array(x_bnds_values[-1]))

        y_bnds = cv[xy_bnds_names[1]]
        self.assertEqual((size[1], 2), y_bnds.shape)
        np.testing.assert_almost_equal(y_bnds.values[0], y_bnds_values[0])
        np.testing.assert_almost_equal(y_bnds.values[-1], y_bnds_values[-1])

    @unittest.skipUnless(is_sentinel3_product(olci_path), f'missing OLCI scene {olci_path}')
    def test_from_olci(self):
        src_ds = open_sentinel3_product(olci_path, {'Oa06_radiance', 'Oa13_radiance', 'Oa20_radiance'})
        src_ds.longitude.load()
        src_ds.latitude.load()

        output_geom = ImageGeom.from_dataset(src_ds, xy_names=('longitude', 'latitude'))
        self.assertEqual(True, output_geom.is_geo_crs)
        self.assertEqual(20259, output_geom.width)
        self.assertEqual(7386, output_geom.height)
        self.assertAlmostEqual(-11.918857, output_geom.x_min)
        self.assertAlmostEqual(59.959791, output_geom.y_min)
        self.assertAlmostEqual(0.00181345416, output_geom.xy_res)

    def test_from_dataset(self):
        src_ds = self.new_source_dataset()

        self._assert_image_geom(ImageGeom((4, 4), None, 0.0, 50.0, 2.0),
                                ImageGeom.from_dataset(src_ds))

        self._assert_image_geom(ImageGeom((7, 7), None, 0.0, 50.0, 1.0),
                                ImageGeom.from_dataset(src_ds,
                                                       xy_oversampling=2.0))

        self._assert_image_geom(ImageGeom((8, 8), None, 0.0, 50.0, 1.0),
                                ImageGeom.from_dataset(src_ds,
                                                       ij_denom=4,
                                                       xy_oversampling=2.0))

    def test_from_dataset_antimeridian(self):
        src_ds = self.new_source_dataset_antimeridian()

        self._assert_image_geom(ImageGeom((4, 4), None, 178.0, 50.0, 2.0),
                                ImageGeom.from_dataset(src_ds))

        self._assert_image_geom(ImageGeom((7, 7), None, 178.0, 50.0, 1.0),
                                ImageGeom.from_dataset(src_ds,
                                                       xy_oversampling=2.0))

        self._assert_image_geom(ImageGeom((8, 8), None, 178.0, 50.0, 1.0),
                                ImageGeom.from_dataset(src_ds,
                                                       ij_denom=4,
                                                       xy_oversampling=2.0))

    def test_from_default_new_cube(self):
        self._assert_image_geom(
            ImageGeom((508, 253), None, -179.5, -89.5, 2 ** -.5),
            ImageGeom.from_dataset(xcube.core.new.new_cube()))

    def _assert_image_geom(self,
                           expected: ImageGeom,
                           actual: ImageGeom):
        self.assertEqual(expected.width, actual.width)
        self.assertEqual(expected.height, actual.height)
        self.assertAlmostEqual(actual.x_min, actual.x_min, delta=1e-5)
        self.assertAlmostEqual(actual.y_min, actual.y_min, delta=1e-5)
        self.assertAlmostEqual(actual.xy_res, actual.xy_res, delta=1e-6)
