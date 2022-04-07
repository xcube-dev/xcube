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

from xcube.core.tilegrid import EARTH_CIRCUMFERENCE_WGS84
from xcube.core.tilegrid import TileGrid
from xcube.core.tilegrid import get_num_levels
from xcube.core.tilegrid import get_unit_factor
from xcube.core.tilegrid import subdivide_size


class TileGridTest(unittest.TestCase):

    def test_web_mercator(self):
        tile_grid = TileGrid.new_web_mercator()

        self.assertEqual((1, 1), tile_grid.num_level_zero_tiles)
        self.assertEqual('EPSG:3857', tile_grid.crs_name)
        self.assertEqual(256, tile_grid.tile_size)
        self.assertEqual(EARTH_CIRCUMFERENCE_WGS84, tile_grid.map_width)
        self.assertEqual(EARTH_CIRCUMFERENCE_WGS84, tile_grid.map_height)
        self.assertEqual(None, tile_grid.min_level)
        self.assertEqual(None, tile_grid.max_level)
        self.assertEqual(None, tile_grid.num_levels)
        half = EARTH_CIRCUMFERENCE_WGS84 / 2
        self.assertEqual((-half, -half, half, half), tile_grid.map_extent)
        self.assertEqual((-half, half), tile_grid.map_origin)

    def test_web_mercator_tile_extent(self):
        tile_grid = TileGrid.new_web_mercator()

        half = EARTH_CIRCUMFERENCE_WGS84 / 2

        self.assertEqual((-half, -half, half, half),
                         tile_grid.get_tile_extent(0, 0, 0))

        self.assertEqual((-half, 0.0, 0.0, half),
                         tile_grid.get_tile_extent(0, 0, 1))
        self.assertEqual((0.0, 0.0, half, half),
                         tile_grid.get_tile_extent(1, 0, 1))
        self.assertEqual((-half, -half, 0.0, 0.0),
                         tile_grid.get_tile_extent(0, 1, 1))
        self.assertEqual((0.0, -half, half, 0.0),
                         tile_grid.get_tile_extent(1, 1, 1))

    def test_geographic(self):
        tile_grid = TileGrid.new_geographic()

        self.assertEqual((2, 1), tile_grid.num_level_zero_tiles)
        self.assertEqual('CRS84', tile_grid.crs_name)
        self.assertEqual(256, tile_grid.tile_size)
        self.assertEqual(360, tile_grid.map_width)
        self.assertEqual(180, tile_grid.map_height)
        self.assertEqual(None, tile_grid.min_level)
        self.assertEqual(None, tile_grid.max_level)
        self.assertEqual(None, tile_grid.num_levels)
        self.assertEqual((-180, -90, 180, 90), tile_grid.map_extent)
        self.assertEqual((-180, 90), tile_grid.map_origin)

    def test_derive(self):
        tile_grid = TileGrid.new_geographic()
        derived_tile_grid = tile_grid.derive(min_level=3, max_level=9)
        self.assertIsInstance(derived_tile_grid, TileGrid)
        self.assertIsNot(tile_grid, derived_tile_grid)
        self.assertEqual(3, derived_tile_grid.min_level)
        self.assertEqual(9, derived_tile_grid.max_level)
        self.assertEqual(10, derived_tile_grid.num_levels)

    def test_geographic_resolutions(self):
        tile_grid = TileGrid.new_geographic()

        with self.assertRaises(ValueError) as cm:
            tile_grid.get_resolutions()
        self.assertEqual('max_value must be given', f'{cm.exception}')

        resolutions = tile_grid.get_resolutions(max_level=6)
        self.assertEqual([0.703125,
                          0.3515625,
                          0.17578125,
                          0.087890625,
                          0.0439453125,
                          0.02197265625,
                          0.010986328125],
                         resolutions)

        resolutions = tile_grid.get_resolutions(min_level=4, max_level=6)
        self.assertEqual([0.0439453125,
                          0.02197265625,
                          0.010986328125],
                         resolutions)

        resolutions = tile_grid.derive(max_level=6). \
            get_resolutions()
        self.assertEqual([0.703125,
                          0.3515625,
                          0.17578125,
                          0.087890625,
                          0.0439453125,
                          0.02197265625,
                          0.010986328125],
                         resolutions)

        resolutions = tile_grid.derive(min_level=4, max_level=6). \
            get_resolutions()
        self.assertEqual([0.0439453125,
                          0.02197265625,
                          0.010986328125],
                         resolutions)

    def test_geographic_tile_extent(self):
        tile_grid = TileGrid.new_geographic()

        self.assertEqual((-180, -90, 0, 90),
                         tile_grid.get_tile_extent(0, 0, 0))
        self.assertEqual((0, -90, 180, 90),
                         tile_grid.get_tile_extent(1, 0, 0))

        self.assertEqual((-180, 0, -90, 90),
                         tile_grid.get_tile_extent(0, 0, 1))
        self.assertEqual((-90, 0, 0, 90),
                         tile_grid.get_tile_extent(1, 0, 1))
        self.assertEqual((0, 0, 90, 90),
                         tile_grid.get_tile_extent(2, 0, 1))
        self.assertEqual((90, 0, 180, 90),
                         tile_grid.get_tile_extent(3, 0, 1))
        self.assertEqual((-180, -90, -90, 0),
                         tile_grid.get_tile_extent(0, 1, 1))
        self.assertEqual((-90, -90, 0, 0),
                         tile_grid.get_tile_extent(1, 1, 1))
        self.assertEqual((0, -90, 90, 0),
                         tile_grid.get_tile_extent(2, 1, 1))
        self.assertEqual((90, -90, 180, 0),
                         tile_grid.get_tile_extent(3, 1, 1))

    def test_web_mercator_get_dataset_level(self):
        tile_grid = TileGrid.new_web_mercator()

        # # This useful for debugging failing tests
        resolutions = tile_grid.get_resolutions(unit_name='meter',
                                                max_level=20)
        for level, res in enumerate(resolutions):
            print(level, res)

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

    def test_web_mercator_get_level_range_for_dataset(self):
        tile_grid = TileGrid.new_web_mercator()

        self.assertEqual(
            (10, 14),
            tile_grid.get_level_range_for_dataset(
                [10, 20, 40, 80, 160],
                'meter'
            )
        )

    def test_geographic_get_dataset_level(self):
        tile_grid = TileGrid.new_geographic()

        # # This useful for debugging failing tests
        resolutions = tile_grid.get_resolutions(unit_name='degree',
                                                max_level=6)
        for level, res in enumerate(resolutions):
            print(level, res)

        num_ds_levels = 6
        ds_resolutions = [
            180 / 256 / 2 ** i for i in reversed(range(num_ds_levels))
        ]

        args = ds_resolutions, 'degree'

        self.assertEqual(5, tile_grid.get_dataset_level(0, *args))
        self.assertEqual(4, tile_grid.get_dataset_level(1, *args))
        self.assertEqual(3, tile_grid.get_dataset_level(2, *args))
        self.assertEqual(2, tile_grid.get_dataset_level(3, *args))
        self.assertEqual(1, tile_grid.get_dataset_level(4, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(5, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(6, *args))
        self.assertEqual(0, tile_grid.get_dataset_level(7, *args))

    def test_geographic_get_level_range_for_dataset(self):
        tile_grid = TileGrid.new_geographic()

        num_ds_levels = 10
        ds_resolutions = [
            180 / 256 / 2 ** i for i in reversed(range(num_ds_levels))
        ]

        self.assertEqual(
            (0, num_ds_levels - 1),
            tile_grid.get_level_range_for_dataset(
                ds_resolutions,
                'degrees'
            )
        )


class TileGridHelpersTest(unittest.TestCase):

    def test_subdivide_size(self):
        self.assertEqual(
            [(2048, 1024), (1024, 512), (512, 256)],
            subdivide_size((2048, 1024), (256, 256))
        )
        self.assertEqual(
            [(360, 180)],
            subdivide_size((360, 180), (256, 256))
        )
        self.assertEqual(
            [(7200, 3600), (3600, 1800), (1800, 900), (900, 450), (450, 225)],
            subdivide_size((7200, 3600), (256, 256))
        )

    def test_get_num_levels(self):
        self.assertEqual(3, get_num_levels((2048, 1024), (256, 256)))
        self.assertEqual(1, get_num_levels((360, 180), (256, 256)))
        self.assertEqual(5, get_num_levels((7200, 3600), (256, 256)))

    def test_get_unit_factor(self):
        self.assertAlmostEqual(1, get_unit_factor('m', 'm'))
        self.assertAlmostEqual(1, get_unit_factor('meters', 'm'))
        self.assertAlmostEqual(1, get_unit_factor('deg', 'degree'))
        self.assertAlmostEqual(111319.49079327358,
                               get_unit_factor('deg', 'm'))
        self.assertAlmostEqual(8.983152841195214e-06,
                               get_unit_factor('m', 'degree'))
        with self.assertRaises(ValueError):
            get_unit_factor('cm', 'm')
        with self.assertRaises(ValueError):
            get_unit_factor('m', 'mdeg')
