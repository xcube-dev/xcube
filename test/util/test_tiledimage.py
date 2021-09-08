from unittest import TestCase

import numpy as np

from xcube.util.tiledimage import OpImage, TransformArrayImage, trim_tile, ArrayImage


class MyTiledImage(OpImage):
    def __init__(self, size, tile_size):
        super().__init__(size, tile_size, (size[0] // tile_size[0], size[1] // tile_size[1]),
                         mode='int32', format='ndarray')

    def compute_tile(self, tile_x, tile_y, rectangle):
        w, h = self.size
        x, y, tw, th = rectangle
        fill_value = float(x + y * w) / float(w * h)
        return np.full((th, tw), fill_value, np.float32)


class TransformArrayImageTest(TestCase):
    def test_default(self):
        a = np.arange(0, 24, dtype=np.int32).reshape((4, 6))
        source_image = ArrayImage(a, (2, 2))
        target_image = TransformArrayImage(source_image)

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

    def test_flip_y(self):
        a = np.arange(0, 24, dtype=np.int32).reshape((4, 6))
        source_image = ArrayImage(a, (2, 2))
        target_image = TransformArrayImage(source_image, flip_y=True)

        self.assertEqual(target_image.size, (6, 4))
        self.assertEqual(target_image.tile_size, (2, 2))
        self.assertEqual(target_image.num_tiles, (3, 2))

        self.assertEqual(target_image.get_tile(0, 0).tolist(), [[18, 19],
                                                                [12, 13]])
        self.assertEqual(target_image.get_tile(1, 0).tolist(), [[20, 21],
                                                                [14, 15]])
        self.assertEqual(target_image.get_tile(2, 0).tolist(), [[22, 23],
                                                                [16, 17]])

        self.assertEqual(target_image.get_tile(0, 1).tolist(), [[6, 7],
                                                                [0, 1]])
        self.assertEqual(target_image.get_tile(1, 1).tolist(), [[8, 9],
                                                                [2, 3]])
        self.assertEqual(target_image.get_tile(2, 1).tolist(), [[10, 11],
                                                                [4, 5]])

    def test_force_2d(self):
        a = np.arange(0, 48, dtype=np.int32).reshape((2, 4, 6))
        source_image = ArrayImage(a[0], (2, 2))
        target_image = TransformArrayImage(source_image, force_2d=True)

        self.assertEqual(target_image.size, (6, 4))
        self.assertEqual(target_image.tile_size, (2, 2))
        self.assertEqual(target_image.num_tiles, (3, 2))


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
