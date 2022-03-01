from unittest import TestCase

import numpy as np
import PIL.Image

from xcube.util.tiledimage import SourceArrayImage
from xcube.util.tiledimage import ColorMappedRgbaImage
from xcube.util.tiledimage import DirectRgbaImage
from xcube.util.tiledimage import OpImage
from xcube.util.tiledimage import NormalizeArrayImage
from xcube.util.tiledimage import trim_tile


class MyTiledImage(OpImage):
    def __init__(self, size, tile_size):
        super().__init__(size, tile_size, (size[0] // tile_size[0], size[1] // tile_size[1]),
                         mode='int32', format='ndarray')

    def compute_tile(self, tile_x, tile_y, rectangle):
        w, h = self.size
        x, y, tw, th = rectangle
        fill_value = float(x + y * w) / float(w * h)
        return np.full((th, tw), fill_value, np.float32)


class ColorMappedRgbaImageTest(TestCase):
    def test_default(self):
        a = np.linspace(0, 255, 24, dtype=np.int32).reshape((4, 6))
        source_image = SourceArrayImage(a, (2, 2))
        cm_rgb_image = ColorMappedRgbaImage(source_image, format='PNG')
        self.assertEqual(cm_rgb_image.size, (6, 4))
        self.assertEqual(cm_rgb_image.tile_size, (2, 2))
        self.assertEqual(cm_rgb_image.num_tiles, (3, 2))
        tile = cm_rgb_image.get_tile(0, 0)
        self.assertIsInstance(tile, PIL.Image.Image)


class DirectRgbaImageTest(TestCase):
    def test_default(self):
        a = np.linspace(0, 255, 24, dtype=np.int32).reshape((4, 6))
        b = np.linspace(255, 0, 24, dtype=np.int32).reshape((4, 6))
        c = np.linspace(50, 200, 24, dtype=np.int32).reshape((4, 6))
        source_images = [
            SourceArrayImage(a, (2, 2)),
            SourceArrayImage(b, (2, 2)),
            SourceArrayImage(c, (2, 2)),
        ]
        cm_rgb_image = DirectRgbaImage(source_images, format='PNG')
        self.assertEqual(cm_rgb_image.size, (6, 4))
        self.assertEqual(cm_rgb_image.tile_size, (2, 2))
        self.assertEqual(cm_rgb_image.num_tiles, (3, 2))
        tile = cm_rgb_image.get_tile(0, 0)
        self.assertIsInstance(tile, PIL.Image.Image)


class NormalizeArrayImageTest(TestCase):
    def test_default(self):
        a = np.arange(0, 24, dtype=np.int32).reshape((4, 6))
        source_image = SourceArrayImage(a, (2, 2))
        target_image = NormalizeArrayImage(source_image)

        self.assertEqual(target_image.size, (6, 4))
        self.assertEqual(target_image.tile_size, (2, 2))
        self.assertEqual(target_image.num_tiles, (3, 2))

        self.assertEqual(target_image.get_tile(0, 0).tolist(), [[0, 1],
                                                                [6, 7]])
        self.assertEqual(target_image.get_tile(1, 0).tolist(), [[2, 3],
                                                                [8, 9]])
        self.assertEqual(target_image.get_tile(2, 0).tolist(), [[4, 5],
                                                                [10, 11]])

        self.assertEqual(target_image.get_tile(0, 1).tolist(), [[12, 13],
                                                                [18, 19]])
        self.assertEqual(target_image.get_tile(1, 1).tolist(), [[14, 15],
                                                                [20, 21]])
        self.assertEqual(target_image.get_tile(2, 1).tolist(), [[16, 17],
                                                                [22, 23]])

    def test_force_2d(self):
        a = np.arange(0, 48, dtype=np.int32).reshape((2, 4, 6))
        source_image = SourceArrayImage(a[0], (2, 2))
        target_image = NormalizeArrayImage(source_image, force_2d=True)

        self.assertEqual(target_image.size, (6, 4))
        self.assertEqual(target_image.tile_size, (2, 2))
        self.assertEqual(target_image.num_tiles, (3, 2))


class SourceArrayImageTest(TestCase):

    def test_flip_y_tiled_evenly(self):
        a = np.arange(0, 24, dtype=np.int32).reshape((4, 6))
        source_array_image = SourceArrayImage(a, (2, 2), flip_y=True)

        self.assertEqual(source_array_image.size, (6, 4))
        self.assertEqual(source_array_image.tile_size, (2, 2))
        self.assertEqual(source_array_image.num_tiles, (3, 2))

        self.assertEqual(source_array_image.get_tile(0, 0).tolist(),
                         [[18, 19],
                          [12, 13]])
        self.assertEqual(source_array_image.get_tile(1, 0).tolist(),
                         [[20, 21],
                          [14, 15]])
        self.assertEqual(source_array_image.get_tile(2, 0).tolist(),
                         [[22, 23],
                          [16, 17]])

        self.assertEqual(source_array_image.get_tile(0, 1).tolist(),
                         [[6, 7],
                          [0, 1]])
        self.assertEqual(source_array_image.get_tile(1, 1).tolist(),
                         [[8, 9],
                          [2, 3]])
        self.assertEqual(source_array_image.get_tile(2, 1).tolist(),
                         [[10, 11],
                          [4, 5]])

    def test_flip_y_not_tiled_evenly(self):
        a = np.arange(0, 30, dtype=np.int32).reshape((5, 6))
        source_array_image = SourceArrayImage(a, (2, 2), flip_y=True)

        self.assertEqual(source_array_image.size, (6, 5))
        self.assertEqual(source_array_image.tile_size, (2, 2))
        self.assertEqual(source_array_image.num_tiles, (3, 3))

        self.assertEqual(source_array_image.get_tile(0, 0).tolist(),
                         [[24, 25],
                          [18, 19]])
        self.assertEqual(source_array_image.get_tile(1, 0).tolist(),
                         [[26, 27],
                          [20, 21]])
        self.assertEqual(source_array_image.get_tile(2, 0).tolist(),
                         [[28, 29],
                          [22, 23]])

        self.assertEqual(source_array_image.get_tile(0, 1).tolist(),
                         [[12, 13],
                          [6, 7]])
        self.assertEqual(source_array_image.get_tile(1, 1).tolist(),
                         [[14, 15],
                          [8, 9]])
        self.assertEqual(source_array_image.get_tile(2, 1).tolist(),
                         [[16, 17],
                          [10, 11]])

        tile_0_2 = source_array_image.get_tile(0, 2).tolist()
        self.assertEqual(tile_0_2[0], [0, 1])
        self.assertTrue(np.isnan(tile_0_2[1][0]))
        self.assertTrue(np.isnan(tile_0_2[1][1]))
        tile_1_2 = source_array_image.get_tile(1, 2).tolist()
        self.assertEqual(tile_1_2[0], [2, 3])
        self.assertTrue(tile_1_2[1][0])
        self.assertTrue(tile_1_2[1][1])
        tile_2_2 = source_array_image.get_tile(2, 2).tolist()
        self.assertEqual(tile_2_2[0], [4, 5])
        self.assertTrue(tile_2_2[1][0])
        self.assertTrue(tile_2_2[1][1])


class TrimTileTest(TestCase):
    # noinspection PyMethodMayBeStatic
    def test_trim_tile(self):
        a = np.arange(0, 6, dtype=np.float32)
        a.shape = 2, 3
        b = trim_tile(a, (3, 2))
        np.testing.assert_equal(b, np.array([[0., 1., 2.],
                                             [3., 4., 5.]]))
        b = trim_tile(a, (4, 2))
        np.testing.assert_equal(b, np.array([[0., 1., 2., np.nan],
                                             [3., 4., 5., np.nan]]))
        b = trim_tile(a, (3, 3))
        np.testing.assert_equal(b, np.array([[0., 1., 2.],
                                             [3., 4., 5.],
                                             [np.nan, np.nan, np.nan]]))
        b = trim_tile(a, (4, 3))
        np.testing.assert_equal(b, np.array([[0., 1., 2., np.nan],
                                             [3., 4., 5., np.nan],
                                             [np.nan, np.nan, np.nan, np.nan]]))
