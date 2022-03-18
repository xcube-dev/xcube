# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import unittest

from xcube.util.tilegrid2 import EARTH_CIRCUMFERENCE_WGS84
from xcube.util.tilegrid2 import TileGrid2


class TileGridTest(unittest.TestCase):

    def test_web_mercator(self):
        tile_grid = TileGrid2.new_web_mercator()

        self.assertEqual((1, 1), tile_grid.num_level_zero_tiles)
        self.assertEqual('EPSG:3857', tile_grid.crs_name)
        self.assertEqual(256, tile_grid.tile_size)
        self.assertEqual(EARTH_CIRCUMFERENCE_WGS84, tile_grid.map_height)
        self.assertEqual(None, tile_grid.map_levels)
        self.assertEqual(None, tile_grid.map_resolutions)

    def test_web_mercator_bbox(self):
        tile_grid = TileGrid2.new_web_mercator()

        half = EARTH_CIRCUMFERENCE_WGS84 / 2

        self.assertEqual((-half, -half, half, half),
                         tile_grid.get_tile_bbox(0, 0, 0))

        self.assertEqual((-half, 0.0, 0.0, half),
                         tile_grid.get_tile_bbox(0, 0, 1))
        self.assertEqual((0.0, 0.0, half, half),
                         tile_grid.get_tile_bbox(1, 0, 1))
        self.assertEqual((-half, -half, 0.0, 0.0),
                         tile_grid.get_tile_bbox(0, 1, 1))
        self.assertEqual((0.0, -half, half, 0.0),
                         tile_grid.get_tile_bbox(1, 1, 1))

    def test_geographic(self):
        tile_grid = TileGrid2.new_geographic()

        self.assertEqual((2, 1), tile_grid.num_level_zero_tiles)
        self.assertEqual('EPSG:4326', tile_grid.crs_name)
        self.assertEqual(256, tile_grid.tile_size)
        self.assertEqual(180, tile_grid.map_height)
        self.assertEqual(None, tile_grid.map_levels)
        self.assertEqual(None, tile_grid.map_resolutions)

    def test_geographic_bbox(self):
        tile_grid = TileGrid2.new_geographic()

        self.assertEqual((-180, -90, 0, 90), tile_grid.get_tile_bbox(0, 0, 0))
        self.assertEqual((0, -90, 180, 90), tile_grid.get_tile_bbox(1, 0, 0))

        self.assertEqual((-180, 0, -90, 90), tile_grid.get_tile_bbox(0, 0, 1))
        self.assertEqual((-90, 0, 0, 90), tile_grid.get_tile_bbox(1, 0, 1))
        self.assertEqual((0, 0, 90, 90), tile_grid.get_tile_bbox(2, 0, 1))
        self.assertEqual((90, 0, 180, 90), tile_grid.get_tile_bbox(3, 0, 1))
        self.assertEqual((-180, -90, -90, 0),
                         tile_grid.get_tile_bbox(0, 1, 1))
        self.assertEqual((-90, -90, 0, 0), tile_grid.get_tile_bbox(1, 1, 1))
        self.assertEqual((0, -90, 90, 0), tile_grid.get_tile_bbox(2, 1, 1))
        self.assertEqual((90, -90, 180, 0), tile_grid.get_tile_bbox(3, 1, 1))

    def test_geographic_dataset_level(self):
        tile_grid = TileGrid2.new_web_mercator()

        for level, res in enumerate(tile_grid.resolutions('meter')):
            print(level, res)
            if level > 16:
                break

        args = [10, 20, 40, 80, 160], 'meter'

        self.assertEqual(4, tile_grid.get_dataset_level(0, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(1, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(2, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(3, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(4, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(5, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(6, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(7, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(8, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(9, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(10, *args))
        self.assertEqual(3, tile_grid.get_dataset_level(11, *args))
        self.assertEqual(2, tile_grid.get_dataset_level(12, *args))
        self.assertEqual(1, tile_grid.get_dataset_level(13, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(14, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(15, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(16, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(17, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(18, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(19, *args))
