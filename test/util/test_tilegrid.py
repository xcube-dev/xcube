from unittest import TestCase

from xcube.util.tilegrid import GLOBAL_GEO_EXTENT
from xcube.util.tilegrid import TileGrid, pow2_1d_subdivisions, pow2_2d_subdivision


class TilingSchemeTest(TestCase):
    def test_to_json(self):
        ts = TileGrid(4, 2, 1, 540, 540, GLOBAL_GEO_EXTENT, inv_y=True)
        self.assertEqual(ts.to_json(), {
            'numLevels': 4,
            'numLevelZeroTilesX': 2,
            'numLevelZeroTilesY': 1,
            'tileHeight': 540,
            'tileWidth': 540,
            'invY': True,
            'extent': {'west': -180.,
                       'south': -90.,
                       'east': 180.,
                       'north': 90.,
                       },
        })

    def test_repr(self):
        ts = TileGrid(4, 2, 1, 540, 540, GLOBAL_GEO_EXTENT, inv_y=False)
        self.assertEqual(repr(ts), 'TileGrid(4, 2, 1, 540, 540, (-180.0, -90.0, 180.0, 90.0), inv_y=False)')
        ts = TileGrid(4, 2, 1, 540, 540, GLOBAL_GEO_EXTENT, inv_y=True)
        self.assertEqual(repr(ts), 'TileGrid(4, 2, 1, 540, 540, (-180.0, -90.0, 180.0, 90.0), inv_y=True)')

    def test_str(self):
        ts = TileGrid(4, 2, 1, 540, 540, GLOBAL_GEO_EXTENT, inv_y=False)
        self.assertEqual(str(ts),
                         'number of pyramid levels: 4\n'
                         'number of tiles at level zero: 2 x 1\n'
                         'pyramid tile size: 540 x 540\n'
                         'image size at level zero: 1080 x 540\n'
                         'image size at level 3: 8640 x 4320\n'
                         'geographic extent: (-180.0, -90.0, 180.0, 90.0)\n'
                         'y-axis points down: yes')

    def test_width_and_height(self):
        ts = TileGrid(4, 2, 1, 540, 540, GLOBAL_GEO_EXTENT, inv_y=False)
        self.assertEqual(ts.width(2), 4320)
        self.assertEqual(ts.height(2), 2160)
        self.assertEqual(ts.max_width, 8640)
        self.assertEqual(ts.max_height, 4320)
        self.assertEqual(ts.min_width, 1080)
        self.assertEqual(ts.min_height, 540)

    def test_num_tiles(self):
        ts = TileGrid(4, 2, 1, 540, 540, GLOBAL_GEO_EXTENT, inv_y=False)
        self.assertEqual(ts.num_tiles_x(0), 2)
        self.assertEqual(ts.num_tiles_y(0), 1)
        self.assertEqual(ts.num_tiles_x(3), 16)
        self.assertEqual(ts.num_tiles_y(3), 8)

    def test_create_cci_ecv(self):
        # 72, 8, 85, 17
        # Soilmoisture CCI - daily L3S
        self.assertEqual(TileGrid.create(1440, 720, 500, 500, GLOBAL_GEO_EXTENT, inv_y=False),
                         TileGrid(2, 2, 1, 360, 360, GLOBAL_GEO_EXTENT, inv_y=False))
        # Aerosol CCI - monthly
        self.assertEqual(TileGrid.create(7200, 3600, 500, 500, GLOBAL_GEO_EXTENT, inv_y=False),
                         TileGrid(4, 2, 1, 450, 450, GLOBAL_GEO_EXTENT, inv_y=False))
        # Cloud CCI - monthly
        self.assertEqual(TileGrid.create(720, 360, 500, 500, GLOBAL_GEO_EXTENT, inv_y=True),
                         TileGrid(1, 2, 1, 360, 360, GLOBAL_GEO_EXTENT, inv_y=True))
        # SST CCI - daily L4
        self.assertEqual(TileGrid.create(8640, 4320, 500, 500, GLOBAL_GEO_EXTENT, inv_y=True),
                         TileGrid(4, 2, 1, 540, 540, GLOBAL_GEO_EXTENT, inv_y=True))
        # Land Cover CCI
        self.assertEqual(TileGrid.create(129600, 64800, 500, 500, GLOBAL_GEO_EXTENT, inv_y=False),
                         TileGrid(6, 6, 3, 675, 675, GLOBAL_GEO_EXTENT, inv_y=False))

    def test_create_cci_ecv_subsets(self):
        # Soilmoisture CCI - daily L3S - use case #6
        self.assertEqual(TileGrid.create(52, 36, 500, 500, (72, 8, 85, 17)),
                         TileGrid(1, 1, 1, 52, 36, (72., 8., 85., 17.)))

    def test_create_subsets(self):
        self.assertEqual(TileGrid.create(4000, 3000, 500, 500, (-20., 10., 60., 70.), inv_y=True),
                         TileGrid(4, 1, 1, 500, 375, (-20., 10., 60., 70.), inv_y=True))
        self.assertEqual(TileGrid.create(4012, 3009, 500, 500, (-20., 10., 60., 70.), inv_y=True),
                         TileGrid(2, 3, 5, 669, 301,
                                  (-20.0, 9.980059820538386, 60.03988035892323, 70.), inv_y=True))
        self.assertEqual(TileGrid.create(4000, 3000, 500, 500, (170., 10., -160., 70.), inv_y=True),
                         TileGrid(4, 1, 1, 500, 375, (170.0, 10.0, -160.0, 70.0), inv_y=True))

    def test_illegal_init(self):
        with self.assertRaises(ValueError):
            TileGrid(0, 2, 1, 540, 540, (0.0, 80., 20.0, 90.0), inv_y=True)
        with self.assertRaises(ValueError):
            TileGrid(4, 0, 1, 540, 540, (0.0, 80., 20.0, 90.0), inv_y=True)
        with self.assertRaises(ValueError):
            TileGrid(4, 2, 0, 540, 540, (0.0, 80., 20.0, 90.0), inv_y=True)
        with self.assertRaises(ValueError):
            TileGrid(4, 2, 1, 0, 540, (0.0, 80., 20.0, 90.0), inv_y=True)
        with self.assertRaises(ValueError):
            TileGrid(4, 2, 1, 540, 0, (0.0, 80., 20.0, 90.0), inv_y=True)
        with self.assertRaises(ValueError):
            TileGrid(4, 2, 1, 540, 540, (0.0, 80., 20.0, 90.01), inv_y=True)

    def test_create_illegal_geo_extent(self):
        # legal - explains why the next must fail
        self.assertEqual(TileGrid.create(50, 25, 5, 5, (0.0, 77.5, 25.0, 90.0), inv_y=True),
                         TileGrid(2, 5, 2, 5, 7, (0.0, 76.0, 25.0, 90.0), inv_y=True))
        with self.assertRaises(ValueError):
            TileGrid.create(50, 25, 5, 5, (0.0, 77.5, 25.0, 90.0), inv_y=False)

        # legal - explains why the next must fail
        self.assertEqual(TileGrid.create(50, 25, 5, 5, (0., -90.0, 25., -77.5), inv_y=False),
                         TileGrid(2, 5, 2, 5, 7, (0.0, -90.0, 25.0, -76.0), inv_y=False))
        with self.assertRaises(ValueError):
            TileGrid.create(50, 25, 5, 5, (0., -90.0, 25., -77.5), inv_y=True)


class Subdivision2DTest(TestCase):
    def test_pow2_2d_subdivision_obvious(self):
        # Aerosol CCI - monthly
        self.assertEqual(pow2_2d_subdivision(360, 180), ((360, 180), (360, 180), (1, 1), 1))
        # Cloud CCI - monthly
        self.assertEqual(pow2_2d_subdivision(720, 360), ((720, 360), (360, 360), (2, 1), 1))
        # SST CCI - daily L4
        self.assertEqual(pow2_2d_subdivision(7200, 3600), ((7200, 3600), (225, 225), (2, 1), 5))
        # OD CCI - monthly L3S
        self.assertEqual(pow2_2d_subdivision(8640, 4320), ((8640, 4320), (270, 270), (2, 1), 5))
        self.assertEqual(pow2_2d_subdivision(8640, 4320, tw_opt=1440, th_opt=1440),
                         ((8640, 4320), (1080, 1080), (2, 1), 3))
        # Land Cover CCI
        self.assertEqual(pow2_2d_subdivision(129600, 64800), ((129600, 64800), (675, 675), (6, 3), 6))

    def test_pow2_2d_subdivision_non_obvious(self):
        self.assertEqual(pow2_2d_subdivision(4823, 5221),
                         ((4823, 5221), (4823, 5221), (1, 1), 1))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=1, h_mode=-1),
                         ((4824, 4180), (603, 1045), (2, 1), 3))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=-1, h_mode=1),
                         ((3860, 5222), (965, 373), (2, 7), 2))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=-1, h_mode=-1),
                         ((3860, 4180), (965, 1045), (1, 1), 3))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=1, h_mode=1),
                         ((4824, 5222), (603, 373), (4, 7), 2))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=1, h_mode=1, tw_opt=500, th_opt=500),
                         ((4824, 5222), (603, 373), (4, 7), 2))

        self.assertEqual(pow2_2d_subdivision(934327, 38294, w_mode=1, h_mode=1, tw_opt=500, th_opt=500),
                         ((934400, 38304), (365, 399), (80, 3), 6))


class Subdivision1DTest(TestCase):
    def test_size_subdivisions(self):
        # Aerosol CCI - monthly
        self.assertEqual(pow2_1d_subdivisions(360),
                         [(360, 360, 1, 1)])
        self.assertEqual(pow2_1d_subdivisions(180),
                         [(180, 180, 1, 1)])
        # Cloud CCI - monthly
        self.assertEqual(pow2_1d_subdivisions(720),
                         [(720, 360, 1, 2)])
        self.assertEqual(pow2_1d_subdivisions(360),
                         [(360, 360, 1, 1)])
        # SST CCI - daily L4
        self.assertEqual(pow2_1d_subdivisions(7200),
                         [(7200, 225, 1, 6),
                          (7200, 450, 1, 5),
                          (7200, 900, 1, 4),
                          (7200, 225, 2, 5),
                          (7200, 450, 2, 4),
                          (7200, 900, 2, 3),
                          (7200, 300, 3, 4),
                          (7200, 600, 3, 3),
                          (7200, 1200, 3, 2),
                          (7200, 225, 4, 4),
                          (7200, 450, 4, 3),
                          (7200, 900, 4, 2),
                          (7200, 360, 5, 3),
                          (7200, 720, 5, 2),
                          (7200, 300, 6, 3),
                          (7200, 600, 6, 2)])
        self.assertEqual(pow2_1d_subdivisions(3600),
                         [(3600, 225, 1, 5),
                          (3600, 450, 1, 4),
                          (3600, 900, 1, 3),
                          (3600, 225, 2, 4),
                          (3600, 450, 2, 3),
                          (3600, 900, 2, 2),
                          (3600, 300, 3, 3),
                          (3600, 600, 3, 2),
                          (3600, 225, 4, 3),
                          (3600, 450, 4, 2),
                          (3600, 360, 5, 2),
                          (3600, 300, 6, 2)])
        # Land Cover CCI
        self.assertEqual(pow2_1d_subdivisions(129600),
                         [(129600, 675, 3, 7),
                          (129600, 405, 5, 7),
                          (129600, 810, 5, 6),
                          (129600, 675, 6, 6)])
        self.assertEqual(pow2_1d_subdivisions(64800),
                         [(64800, 675, 3, 6),
                          (64800, 405, 5, 6),
                          (64800, 810, 5, 5),
                          (64800, 675, 6, 5)])

    def test_pow2_1d_subdivision_illegal(self):
        with self.assertRaises(ValueError):
            pow2_1d_subdivisions(-100)

        with self.assertRaises(ValueError):
            pow2_1d_subdivisions(100, ts_min=-100)

        with self.assertRaises(ValueError):
            pow2_1d_subdivisions(100, ts_opt=0)

        with self.assertRaises(ValueError):
            pow2_1d_subdivisions(100, nt0_max=-1)

        with self.assertRaises(ValueError):
            pow2_1d_subdivisions(100, nl_max=-1)
