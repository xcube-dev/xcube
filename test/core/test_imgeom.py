import unittest

import numpy as np

from xcube.core.rectify import ImageGeom
from xcube.core.sentinel3 import is_sentinel3_product
from xcube.core.sentinel3 import open_sentinel3_product
from .test_geocoding import SourceDatasetMixin

olci_path = 'C:\\Users\\Norman\\Downloads\\S3B_OL_1_EFR____20190728T103451_20190728T103751_20190729T141105_0179_028_108_1800_LN1_O_NT_002.SEN3'


class ImageGeomTest(SourceDatasetMixin, unittest.TestCase):
    def test_size(self):
        image_geom = ImageGeom((3600, 1800))
        self.assertEqual((3600, 1800), image_geom.size)

        image_geom = ImageGeom(3600)
        self.assertEqual((3600, 3600), image_geom.size)

        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            ImageGeom(None)

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            ImageGeom((3600, 1800, 4))

    def test_tile_size(self):
        image_geom = ImageGeom(size=(3600, 1800))
        self.assertEqual((3600, 1800), image_geom.size)
        self.assertEqual((3600, 1800), image_geom.tile_size)
        self.assertEqual(False, image_geom.is_tiled)

        image_geom = ImageGeom(size=(3600, 1800), tile_size=None)
        self.assertEqual((3600, 1800), image_geom.size)
        self.assertEqual((3600, 1800), image_geom.tile_size)
        self.assertEqual(False, image_geom.is_tiled)

        image_geom = ImageGeom(size=(3600, 1800), tile_size=(512, 256))
        self.assertEqual((3600, 1800), image_geom.size)
        self.assertEqual((512, 256), image_geom.tile_size)
        self.assertEqual(True, image_geom.is_tiled)

        image_geom = ImageGeom(size=(3600, 1800), tile_size=270)
        self.assertEqual((3600, 1800), image_geom.size)
        self.assertEqual((270, 270), image_geom.tile_size)
        self.assertEqual(True, image_geom.is_tiled)

        image_geom = ImageGeom(size=(360, 180), tile_size=(512, 256))
        self.assertEqual((360, 180), image_geom.size)
        self.assertEqual((360, 180), image_geom.tile_size)
        self.assertEqual(False, image_geom.is_tiled)

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            ImageGeom((3600, 1800), ((512,)))

    def test_is_crossing_antimeridian(self):
        output_geom = ImageGeom(size=(13, 13), x_min=0.0, y_min=+50.0, xy_res=0.5)
        self.assertFalse(output_geom.is_crossing_antimeridian)

        output_geom = ImageGeom(size=(13, 13), x_min=178.0, y_min=+50.0, xy_res=0.5)
        self.assertTrue(output_geom.is_crossing_antimeridian)

    def test_derive(self):
        image_geom = ImageGeom((2048, 1024))
        self.assertEqual((2048, 1024), image_geom.tile_size)
        new_image_geom = image_geom.derive(tile_size=512)
        self.assertIsNot(new_image_geom, image_geom)
        self.assertEqual((2048, 1024), new_image_geom.size)
        self.assertEqual((512, 512), new_image_geom.tile_size)

    def test_tile_bboxes(self):
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

    def test_tile_xy_bboxes(self):
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

    @unittest.skipUnless(is_sentinel3_product(olci_path), f'missing OLCI scene {olci_path}')
    def test_from_olci(self):
        src_ds = open_sentinel3_product(olci_path, {'Oa06_radiance', 'Oa13_radiance', 'Oa20_radiance'})
        src_ds.longitude.load()
        src_ds.latitude.load()

        output_geom = ImageGeom.from_dataset(src_ds, xy_names=('longitude', 'latitude'))
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

    def _assert_image_geom(self,
                           expected: ImageGeom,
                           actual: ImageGeom):
        self.assertEqual(expected.width, actual.width)
        self.assertEqual(expected.height, actual.height)
        self.assertAlmostEqual(actual.x_min, actual.x_min, delta=1e-5)
        self.assertAlmostEqual(actual.y_min, actual.y_min, delta=1e-5)
        self.assertAlmostEqual(actual.xy_res, actual.xy_res, delta=1e-6)
