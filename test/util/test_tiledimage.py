from unittest import TestCase

import numpy as np

from xcube.constants import GLOBAL_GEO_EXTENT
from xcube.util.tiledimage import ImagePyramid, OpImage, create_ndarray_downsampling_image, \
    TransformArrayImage, FastNdarrayDownsamplingImage, trim_tile
from xcube.util.tiledimage import downsample_ndarray, aggregate_ndarray_mean, aggregate_ndarray_first
from xcube.util.tilegrid import TileGrid


class MyTiledImage(OpImage):
    def __init__(self, size, tile_size):
        super().__init__(size, tile_size, (size[0] // tile_size[0], size[1] // tile_size[1]),
                         mode='int32', format='ndarray')

    def compute_tile(self, tile_x, tile_y, rectangle):
        w, h = self.size
        x, y, tw, th = rectangle
        fill_value = float(x + y * w) / float(w * h)
        return np.full((th, tw), fill_value, np.float32)


class NdarrayImageTest(TestCase):
    def test_default(self):
        a = np.arange(0, 24, dtype=np.int32)
        a.shape = 4, 6
        source_image = FastNdarrayDownsamplingImage(a, (2, 2), 0)
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
        a = np.arange(0, 24, dtype=np.int32)
        a.shape = 4, 6
        source_image = FastNdarrayDownsamplingImage(a, (2, 2), 0)
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

    def test_level_2(self):
        a = np.arange(0, 16 * 24, dtype=np.int32)
        a.shape = 16, 24
        source_image = FastNdarrayDownsamplingImage(a, (2, 2), 2)
        target_image = TransformArrayImage(source_image)

        self.assertEqual(target_image.size, (6, 4))
        self.assertEqual(target_image.tile_size, (2, 2))
        self.assertEqual(target_image.num_tiles, (3, 2))

        self.assertEqual(target_image.get_tile(0, 0).tolist(), [[0, 4],
                                                                [96, 100]])
        self.assertEqual(target_image.get_tile(1, 0).tolist(), [[8, 12],
                                                                [104, 108]])
        self.assertEqual(target_image.get_tile(2, 0).tolist(), [[16, 20],
                                                                [112, 116]])

        self.assertEqual(target_image.get_tile(0, 1).tolist(), [[192, 196],
                                                                [288, 292]])
        self.assertEqual(target_image.get_tile(1, 1).tolist(), [[200, 204],
                                                                [296, 300]])
        self.assertEqual(target_image.get_tile(2, 1).tolist(), [[208, 212],
                                                                [304, 308]])

    def test_force_masked(self):
        a = np.arange(0, 24, dtype=np.int32)
        a.shape = 4, 6
        source_image = FastNdarrayDownsamplingImage(a, (2, 2), 0)
        target_image = TransformArrayImage(source_image, force_masked=True, no_data_value=14)

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
        self.assertEqual(target_image.get_tile(1, 1).tolist(), [[None, 15],
                                                                [20, 21]])
        self.assertEqual(target_image.get_tile(2, 1).tolist(), [[16, 17],
                                                                [22, 23]])

        a = np.arange(0, 24, dtype=np.float64)
        a.shape = 4, 6
        a[1, 2] = np.nan
        a[2, 5] = np.inf
        source_image = FastNdarrayDownsamplingImage(a, (2, 2), 0)
        target_image = TransformArrayImage(source_image, force_masked=True)

        self.assertEqual(target_image.size, (6, 4))
        self.assertEqual(target_image.num_tiles, (3, 2))

        self.assertEqual(target_image.get_tile(0, 0).tolist(), [[0, 1],
                                                                [6, 7]])
        self.assertEqual(target_image.get_tile(1, 0).tolist(), [[2, 3],
                                                                [None, 9]])
        self.assertEqual(target_image.get_tile(2, 0).tolist(), [[4, 5],
                                                                [10, 11]])

        self.assertEqual(target_image.get_tile(0, 1).tolist(), [[12, 13],
                                                                [18, 19]])
        self.assertEqual(target_image.get_tile(1, 1).tolist(), [[14, 15],
                                                                [20, 21]])
        self.assertEqual(target_image.get_tile(2, 1).tolist(), [[16, None],
                                                                [22, 23]])

    def test_force_2d(self):
        a = np.arange(0, 48, dtype=np.int32)
        a.shape = 2, 4, 6
        source_image = FastNdarrayDownsamplingImage(a, (2, 2), 0)
        target_image = TransformArrayImage(source_image, force_2d=True)

        self.assertEqual(target_image.size, (6, 4))
        self.assertEqual(target_image.tile_size, (2, 2))
        self.assertEqual(target_image.num_tiles, (3, 2))


class TrimTileTest(TestCase):
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


class ImagePyramidTest(TestCase):
    def test_create_from_image(self):
        width = 8640
        height = 4320

        pyramid = ImagePyramid.create_from_image(MyTiledImage((width, height), (270, 270)),
                                                 create_ndarray_downsampling_image,
                                                 aggregator=aggregate_ndarray_mean)

        self.assertEqual((270, 270), pyramid.tile_size)
        self.assertEqual((2, 1), pyramid.num_level_zero_tiles)
        self.assertEqual(5, pyramid.num_levels)

        level_image_0 = pyramid.get_level_image(0)
        self.assertEqual((540, 270), level_image_0.size)
        self.assertEqual((270, 270), level_image_0.tile_size)
        self.assertEqual((2, 1), level_image_0.num_tiles)

        level_image_1 = pyramid.get_level_image(1)
        self.assertEqual((1080, 540), level_image_1.size)
        self.assertEqual((270, 270), level_image_1.tile_size)
        self.assertEqual((4, 2), level_image_1.num_tiles)

        level_image_2 = pyramid.get_level_image(2)
        self.assertEqual((2160, 1080), level_image_2.size)
        self.assertEqual((270, 270), level_image_2.tile_size)
        self.assertEqual((8, 4), level_image_2.num_tiles)

        level_image_3 = pyramid.get_level_image(3)
        self.assertEqual((4320, 2160), level_image_3.size)
        self.assertEqual((270, 270), level_image_3.tile_size)
        self.assertEqual((16, 8), level_image_3.num_tiles)

        level_image_4 = pyramid.get_level_image(4)
        self.assertEqual((8640, 4320), level_image_4.size)
        self.assertEqual((270, 270), level_image_4.tile_size)
        self.assertEqual((32, 16), level_image_4.num_tiles)

        tile_4_0_0 = level_image_4.get_tile(0, 0)
        self.assertEqual((270, 270), tile_4_0_0.shape)
        self.assertEqual(0, tile_4_0_0[0, 0])
        self.assertEqual(0, tile_4_0_0[269, 269])

        tile_4_31_15 = level_image_4.get_tile(31, 15)
        self.assertEqual((270, 270), tile_4_31_15.shape)
        self.assertAlmostEqual(0.93772423, tile_4_31_15[0, 0])
        self.assertAlmostEqual(0.93772423, tile_4_31_15[269, 269])

        tile_3_0_0 = level_image_3.get_tile(0, 0)
        self.assertEqual((270, 270), tile_3_0_0.shape)
        self.assertAlmostEqual(0, tile_3_0_0[0, 0])
        self.assertAlmostEqual(0.06250723, tile_3_0_0[269, 269])

        tile_3_15_7 = level_image_3.get_tile(15, 7)
        self.assertEqual((270, 270), tile_3_0_0.shape)
        self.assertAlmostEqual(0.87521702, tile_3_15_7[0, 0])
        self.assertAlmostEqual(0.93772423, tile_3_15_7[269, 269])

        tile_0_0_0 = level_image_0.get_tile(0, 0)
        self.assertEqual((270, 270), tile_0_0_0.shape)
        self.assertAlmostEqual(0, tile_0_0_0[0, 0])
        self.assertAlmostEqual(0.93760848, tile_0_0_0[269, 269])

        tile_0_1_0 = level_image_0.get_tile(1, 0)
        self.assertEqual((270, 270), tile_0_1_0.shape)
        self.assertAlmostEqual(0.00011574, tile_0_1_0[0, 0])
        self.assertAlmostEqual(0.93772423, tile_0_1_0[269, 269])

    def test_create_from_array(self):
        width = 8640
        height = 4320

        # Typical NetCDF shape: time, lat, lon
        array = np.zeros((1, height, width))

        tiling_scheme = TileGrid.create(width, height, 270, 270, geo_extent=GLOBAL_GEO_EXTENT)
        pyramid = ImagePyramid.create_from_array(array, tiling_scheme)

        self.assertEqual((270, 270), pyramid.tile_size)
        self.assertEqual((2, 1), pyramid.num_level_zero_tiles)
        self.assertEqual(5, pyramid.num_levels)

        level_image_0 = pyramid.get_level_image(0)
        self.assertEqual((540, 270), level_image_0.size)
        self.assertEqual((270, 270), level_image_0.tile_size)
        self.assertEqual((2, 1), level_image_0.num_tiles)

        level_image_1 = pyramid.get_level_image(1)
        self.assertEqual((1080, 540), level_image_1.size)
        self.assertEqual((270, 270), level_image_1.tile_size)
        self.assertEqual((4, 2), level_image_1.num_tiles)

        level_image_2 = pyramid.get_level_image(2)
        self.assertEqual((2160, 1080), level_image_2.size)
        self.assertEqual((270, 270), level_image_2.tile_size)
        self.assertEqual((8, 4), level_image_2.num_tiles)

        level_image_3 = pyramid.get_level_image(3)
        self.assertEqual((4320, 2160), level_image_3.size)
        self.assertEqual((270, 270), level_image_3.tile_size)
        self.assertEqual((16, 8), level_image_3.num_tiles)

        level_image_4 = pyramid.get_level_image(4)
        self.assertEqual((8640, 4320), level_image_4.size)
        self.assertEqual((270, 270), level_image_4.tile_size)
        self.assertEqual((32, 16), level_image_4.num_tiles)

        tile_4_0_0 = level_image_4.get_tile(0, 0)
        self.assertEqual((1, 270, 270), tile_4_0_0.shape)
        self.assertEqual(0, tile_4_0_0[..., 0, 0])
        self.assertEqual(0, tile_4_0_0[..., 269, 269])

        tile_4_31_15 = level_image_4.get_tile(31, 15)
        self.assertEqual((1, 270, 270), tile_4_31_15.shape)
        self.assertAlmostEqual(0, tile_4_31_15[..., 0, 0])
        self.assertAlmostEqual(0, tile_4_31_15[..., 269, 269])

        tile_3_0_0 = level_image_3.get_tile(0, 0)
        self.assertEqual((1, 270, 270), tile_3_0_0.shape)
        self.assertAlmostEqual(0, tile_3_0_0[..., 0, 0])
        self.assertAlmostEqual(0, tile_3_0_0[..., 269, 269])

        tile_3_15_7 = level_image_3.get_tile(15, 7)
        self.assertEqual((1, 270, 270), tile_3_0_0.shape)
        self.assertAlmostEqual(0, tile_3_15_7[..., 0, 0])
        self.assertAlmostEqual(0, tile_3_15_7[..., 269, 269])

        tile_0_0_0 = level_image_0.get_tile(0, 0)
        self.assertEqual((1, 270, 270), tile_0_0_0.shape)
        self.assertAlmostEqual(0, tile_0_0_0[..., 0, 0])
        self.assertAlmostEqual(0, tile_0_0_0[..., 269, 269])

        tile_0_1_0 = level_image_0.get_tile(1, 0)
        self.assertEqual((1, 270, 270), tile_0_1_0.shape)
        self.assertAlmostEqual(0, tile_0_1_0[..., 0, 0])
        self.assertAlmostEqual(0, tile_0_1_0[..., 269, 269])


class ArrayResampleTest(TestCase):
    def test_numpy_reduce(self):
        nan = np.nan
        a = np.zeros((8, 6))
        a[0::2, 0::2] = 1.1
        a[0::2, 1::2] = 2.2
        a[1::2, 0::2] = 3.3
        a[1::2, 1::2] = 4.4
        a[6, 4] = np.nan

        self.assertEqual(a.shape, (8, 6))
        np.testing.assert_equal(a, np.array([[1.1, 2.2, 1.1, 2.2, 1.1, 2.2],
                                             [3.3, 4.4, 3.3, 4.4, 3.3, 4.4],
                                             [1.1, 2.2, 1.1, 2.2, 1.1, 2.2],
                                             [3.3, 4.4, 3.3, 4.4, 3.3, 4.4],
                                             [1.1, 2.2, 1.1, 2.2, 1.1, 2.2],
                                             [3.3, 4.4, 3.3, 4.4, 3.3, 4.4],
                                             [1.1, 2.2, 1.1, 2.2, nan, 2.2],
                                             [3.3, 4.4, 3.3, 4.4, 3.3, 4.4]]))

        b = downsample_ndarray(a)
        self.assertEqual(b.shape, (4, 3))
        np.testing.assert_equal(b, np.array([[2.75, 2.75, 2.75],
                                             [2.75, 2.75, 2.75],
                                             [2.75, 2.75, 2.75],
                                             [2.75, 2.75, nan]]))

        b = downsample_ndarray(a, aggregator=aggregate_ndarray_first)
        self.assertEqual(b.shape, (4, 3))
        np.testing.assert_equal(b, np.array([[1.1, 1.1, 1.1],
                                             [1.1, 1.1, 1.1],
                                             [1.1, 1.1, 1.1],
                                             [1.1, 1.1, nan]]))
