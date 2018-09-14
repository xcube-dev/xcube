import unittest

from xcube.grid.feg import FixedEarthGrid, main, get_tile_counts_and_sizes


class FixedEarthGridTest(unittest.TestCase):
    def test_get_res(self):
        feg = FixedEarthGrid()
        self.assertEqual(feg.get_res(0), 1)
        self.assertEqual(feg.get_res(1), 0.5)
        self.assertEqual(feg.get_res(2), 0.25)
        self.assertEqual(feg.get_res(9, units='meters'), 34.60360785590278)
        self.assertEqual(feg.get_res(10, units='meters'), 17.301803927951389)
        feg = FixedEarthGrid(level_zero_res=0.3)
        self.assertEqual(feg.get_res(7, units='meters'), 41.52432942708333)
        self.assertEqual(feg.get_res(8, units='meters'), 20.762164713541665)
        feg = FixedEarthGrid(level_zero_res=0.2)
        self.assertEqual(feg.get_res(7, units='meters'), 27.682886284722224)
        self.assertEqual(feg.get_res(8, units='meters'), 13.841443142361112)
        feg = FixedEarthGrid(level_zero_res=0.18)
        self.assertEqual(feg.get_res(7, units='meters'), 24.91459765625)
        self.assertEqual(feg.get_res(8, units='meters'), 12.457298828125)
        feg = FixedEarthGrid(level_zero_res=0.05)
        self.assertEqual(feg.get_res(5, units='meters'), 27.682886284722224)
        self.assertEqual(feg.get_res(6, units='meters'), 13.841443142361112)

    def test_get_level(self):
        feg = FixedEarthGrid()
        self.assertEqual(feg.get_level(1.0), 0)
        self.assertEqual(feg.get_level(0.5), 1)
        self.assertEqual(feg.get_level(0.25), 2)
        self.assertEqual(feg.get_level(300, units='meters'), 6)
        feg = FixedEarthGrid(level_zero_res=0.3)
        self.assertEqual(feg.get_level(300, units='meters'), 4)
        feg = FixedEarthGrid(level_zero_res=0.2)
        self.assertEqual(feg.get_level(300, units='meters'), 4)
        feg = FixedEarthGrid(level_zero_res=0.18)
        self.assertEqual(feg.get_level(300, units='meters'), 3)
        feg = FixedEarthGrid(level_zero_res=0.05)
        self.assertEqual(feg.get_level(300, units='meters'), 2)

    def test_get_level_and_res(self):
        feg = FixedEarthGrid()
        self.assertEqual(feg.get_level_and_res(2.0), (0, 1))
        self.assertEqual(feg.get_level_and_res(1.0), (0, 1))
        self.assertEqual(feg.get_level_and_res(0.5), (1, 0.5))
        self.assertEqual(feg.get_level_and_res(0.4), (1, 0.5))
        self.assertEqual(feg.get_level_and_res(0.3), (2, 0.25))
        self.assertEqual(feg.get_level_and_res(0.25), (2, 0.25))
        self.assertEqual(feg.get_level_and_res(0.2), (2, 0.25))
        self.assertEqual(feg.get_level_and_res(0.1), (3, 0.125))
        self.assertEqual(feg.get_level_and_res(0.05), (4, 0.0625))
        self.assertEqual(feg.get_level_and_res(0.03), (5, 0.03125))
        self.assertEqual(feg.get_level_and_res(0.02), (6, 0.015625))
        self.assertEqual(feg.get_level_and_res(0.01), (7, 0.0078125))
        self.assertEqual(feg.get_level_and_res(0.005), (8, 0.00390625))

    def test_get_grid_size(self):
        feg = FixedEarthGrid()
        self.assertEqual(feg.get_grid_size(0), (360, 180))
        self.assertEqual(feg.get_grid_size(1), (720, 360))
        self.assertEqual(feg.get_grid_size(5), (11520, 5760))

        feg = FixedEarthGrid(level_zero_res=0.25)
        self.assertEqual(feg.get_grid_size(0), (1440, 720))
        self.assertEqual(feg.get_grid_size(1), (2880, 1440))
        self.assertEqual(feg.get_grid_size(5), (46080, 23040))


class MainTest(unittest.TestCase):
    def test_main(self):
        # Simple smoke test
        self.assertEqual(0, main(['--help']))
        self.assertEqual(0, main(['--units', 'm', '0', '52', '5', '54', '0.03']))


class GetTileCountsAndSizesTest(unittest.TestCase):

    def test_tile_counts_and_sizes(self):
        resolutions_and_expected_tile_counts_and_sizes = [
            (1 / 1,
             [(1, 180)]),
            (1 / 2,
             [(1, 360)]),
            (1 / 4,
             [(1, 720)]),
            (1 / 5,
             [(2, 450)]),
            (1 / 8,
             [(2, 720), (3, 480)]),
            (1 / 10,
             [(3, 600), (4, 450)]),
            (1 / 20,
             [(5, 720), (6, 600), (8, 450), (9, 400)]),
            (1 / 25,
             [(9, 500), (10, 450), (12, 375)]),
            (1 / 50,
             [(15, 600), (18, 500), (20, 450), (24, 375)]),
            (1 / 100,
             [(25, 720), (30, 600), (36, 500), (40, 450), (45, 400), (48, 375)]),
            (1 / 200,
             [(50, 720), (60, 600), (72, 500), (75, 480), (80, 450), (90, 400), (96, 375)]),
            (1 / 250,
             [(72, 625), (75, 600), (90, 500), (100, 450), (120, 375)]),
            (1 / 500,
             [(125, 720), (144, 625), (150, 600), (180, 500), (200, 450), (225, 400), (240, 375)]),
            (1 / 1000,
             [(250, 720), (288, 625), (300, 600), (360, 500), (375, 480), (400, 450), (450, 400), (480, 375)]),
            (1 / 2000,
             [(500, 720), (576, 625), (600, 600), (625, 576), (720, 500), (750, 480), (800, 450), (900, 400), (
                 960, 375)]),
            (1 / 2500,
             [(625, 720), (720, 625), (750, 600), (900, 500), (1000, 450), (1125, 400), (1200, 375)]),
            (1 / 5000,
             [(1250, 720), (1440, 625), (1500, 600), (1800, 500), (1875, 480), (2000, 450), (2250, 400), (2400, 375)]),
            (1 / 10000,
             [(2500, 720), (2880, 625), (3000, 600), (3125, 576), (3600, 500), (3750, 480), (4000, 450), (
                 4500, 400), (4800, 375)]),
        ]
        for res, expected_tile_counts_and_sizes in resolutions_and_expected_tile_counts_and_sizes:
            actual_tile_counts_and_sizes = get_tile_counts_and_sizes(res)
            self.assertEqual(expected_tile_counts_and_sizes, actual_tile_counts_and_sizes)
