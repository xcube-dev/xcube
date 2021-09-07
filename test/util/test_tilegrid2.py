import unittest
from unittest import TestCase

import pyproj

from xcube.util.tilegrid2 import GeographicTileGrid
from xcube.util.tilegrid2 import ImageTileGrid
from xcube.util.tilegrid2 import tile_grid_to_ol_xyz_source_options


class Crs84TileGridTest(TestCase):
    def test_it(self):
        tg = GeographicTileGrid()
        self.assertEqual((256, 256), tg.tile_size)
        self.assertEqual(None, tg.min_level)
        self.assertEqual(None, tg.max_level)
        self.assertEqual(pyproj.CRS.from_string('CRS84'), tg.crs)
        self.assertEqual((-180, -90, 180, 90), tg.extent)
        self.assertEqual((-180, -90), tg.origin)
        self.assertEqual((512, 256), tg.get_image_size(0))
        self.assertEqual((1024, 512), tg.get_image_size(1))
        self.assertEqual((512 * 2 ** 15, 256 * 2 ** 15), tg.get_image_size(15))
        self.assertEqual((2, 1), tg.get_num_tiles(0))
        self.assertEqual((4, 2), tg.get_num_tiles(1))
        self.assertEqual((2 ** 16, 2 ** 15), tg.get_num_tiles(15))
        self.assertEqual(180 / 256, tg.get_resolution(0))
        self.assertEqual(180 / 512, tg.get_resolution(1))
        self.assertEqual(180 / (256 * 2 ** 15), tg.get_resolution(15))


class ImageTileGridTest(TestCase):
    def test_it(self):
        tg = ImageTileGrid(image_size=(4000, 2000),
                           tile_size=(512, 512),
                           crs=pyproj.CRS.from_string('WGS84'),
                           image_res=0.05,
                           image_origin=(0, 0))
        self.assertEqual((512, 512), tg.tile_size)
        self.assertEqual(None, tg.min_level)
        self.assertEqual(2, tg.max_level)
        self.assertEqual(pyproj.CRS.from_string('WGS84'), tg.crs)
        self.assertEqual((0, 0, 200.0, 100.0), tg.extent)
        self.assertEqual((0, 0), tg.origin)
        self.assertEqual((1000, 500), tg.get_image_size(0))
        self.assertEqual((2000, 1000), tg.get_image_size(1))
        self.assertEqual((4000, 2000), tg.get_image_size(2))
        self.assertEqual((2, 1), tg.get_num_tiles(0))
        self.assertEqual((4, 2), tg.get_num_tiles(1))
        self.assertEqual((8, 4), tg.get_num_tiles(2))
        self.assertEqual(0.2, tg.get_resolution(0))
        self.assertEqual(0.1, tg.get_resolution(1))
        self.assertEqual(0.05, tg.get_resolution(2))


class OpenLayersOptionsTest(unittest.TestCase):
    def test_with_max_level(self):
        tile_grid = ImageTileGrid((7200, 3600),
                                  (512, 512),
                                  'EPSG:4326',
                                  180 / 3600,
                                  (-180, -90))
        self.assertEqual(
            {
                'url': 'https://xcube.server/tile/{x}/{y}/{z}.png',
                'projection': 'EPSG:4326',
                'tileGrid': {
                    'tileSize': [512, 512],
                    'sizes': [[2, 1], [4, 2], [8, 4], [16, 8]],
                    'resolutions': [0.4, 0.2, 0.1, 0.05],
                    'extent': [-180, -90, 180.0, 90.0],
                    'origin': [-180, -90],
                },
            },
            tile_grid_to_ol_xyz_source_options(
                tile_grid,
                'https://xcube.server/tile/{x}/{y}/{z}.png'
            )
        )

    def test_with_wo_max_level(self):
        tile_grid = GeographicTileGrid()
        self.assertEqual(
            {
                'url': 'https://xcube.server/tile/{x}/{y}/{z}.png',
                'projection': 'CRS84',
                'tileSize': [256, 256],
                'maxResolution': 0.703125,
            },
            tile_grid_to_ol_xyz_source_options(
                tile_grid,
                'https://xcube.server/tile/{x}/{y}/{z}.png'
            )
        )
