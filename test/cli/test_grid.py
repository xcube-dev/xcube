import unittest
from fractions import Fraction
from typing import List

import click
import click.testing

from xcube.cli.grid import grid as cli
from xcube.cli.grid import factor_out_two, get_adjusted_box, get_levels, meters_to_degrees, find_close_resolutions


class GridToolsTest(unittest.TestCase):

    def test_factor_out_two(self):
        with self.assertRaises(ValueError):
            factor_out_two(-1)
        self.assertEqual((0, 0), factor_out_two(0))
        self.assertEqual((1, 0), factor_out_two(1))
        self.assertEqual((1, 1), factor_out_two(2))
        self.assertEqual((3, 0), factor_out_two(3))
        self.assertEqual((1, 2), factor_out_two(4))
        self.assertEqual((83743, 0), factor_out_two(83743))
        self.assertEqual((2617, 5), factor_out_two(83744))
        self.assertEqual(83744, 2617 * 2 ** 5)

    def test_get_adjusted_box(self):
        self.assertEqual((0.0, 49.21875, 5.625, 53.4375),
                         get_adjusted_box(0.0, 50.0, 5.0, 52.5, 540 / 384))

    def test_get_levels(self):
        results = get_levels(3, 180, level_min=4)
        self.assertEqual(6, len(results))
        self.assertEqual(['L', 0, 1, 2, 3, 4],
                         [row[0] for row in results])
        self.assertEqual(['H', 3, 6, 12, 24, 48],
                         [row[1] for row in results])
        self.assertEqual(['R', Fraction(60, 1), Fraction(30, 1), Fraction(15, 1), Fraction(15, 2), Fraction(15, 4)],
                         [row[2] for row in results])
        self.assertEqual(['R (deg)', 60.0, 30.0, 15.0, 7.5, 3.75],
                         [row[3] for row in results])
        self.assertEqual(['R (m)', 6679169.45, 3339584.72, 1669792.36, 834896.18, 417448.09],
                         [row[4] for row in results])

    def test_find_close_resolutions(self):
        target_res = meters_to_degrees(300)
        results = find_close_resolutions(target_res, target_res * 1.0 / 100, 180)
        self.assertEqual(43, len(results))
        self.assertEqual(['R_D (%)', 0.012, -0.036, 0.059, -0.084],
                         [row[0] for row in results[0:5]])
        self.assertEqual(['R_NOM', 45, 5, 45, 45],
                         [row[1] for row in results[0:5]])
        self.assertEqual(['R_DEN', 16696, 1856, 16688, 16712],
                         [row[2] for row in results[0:5]])
        self.assertEqual(['R (deg)',
                          0.002695256348826066,
                          0.0026939655172413795,
                          0.002696548418024928,
                          0.0026926759214935376],
                         [row[3] for row in results[0:5]])
        self.assertEqual(['R (m)', 300.03, 299.89, 300.18, 299.75],
                         [row[4] for row in results[0:5]])
        self.assertEqual(['H', 66784, 66816, 66752, 66848],
                         [row[5] for row in results[0:5]])
        self.assertEqual(['H0', 2087, 261, 1043, 2089],
                         [row[6] for row in results[0:5]])


class GridCliTest(unittest.TestCase):

    @classmethod
    def invoke_cli(cls, args: List[str]):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, args)

    def test_levels(self):
        result = self.invoke_cli(["levels", "-R", "1/384"])
        self.assertEqual(("""
L	H	R	R (deg)	R (m)
0	135	4/3	1.3333333333333333	148425.99
1	270	2/3	0.6666666666666666	74212.99
2	540	1/3	0.3333333333333333	37106.5
3	1080	1/6	0.16666666666666666	18553.25
4	2160	1/12	0.08333333333333333	9276.62
5	4320	1/24	0.041666666666666664	4638.31
6	8640	1/48	0.020833333333333332	2319.16
7	17280	1/96	0.010416666666666666	1159.58
8	34560	1/192	0.005208333333333333	579.79
9	69120	1/384	0.0026041666666666665	289.89
"""),
                         result.output)
        self.assertEqual(0, result.exit_code)

    def test_res(self):
        result = self.invoke_cli(["res", "300m", "-D", "1%", "--sort_by", "+H0", "-N", "6"])
        print(result.output)
        self.assertEqual(("""
R_D (%)	R_NOM	R_DEN	R (deg)	R (m)	H	H0	L
0.348	9	3328	0.002704326923076923	301.04	66560	65	10
-0.418	45	16768	0.0026836832061068704	298.75	67072	131	9
0.736	45	16576	0.00271476833976834	302.21	66304	259	8
-0.036	5	1856	0.0026939655172413795	299.89	66816	261	8
-0.797	45	16832	0.0026734790874524714	297.61	67328	263	8
37 more...
"""),
                         result.output)
        self.assertEqual(0, result.exit_code)

    def test_abox(self):
        result = self.invoke_cli(['abox', "0,50,5,52.5", "-R", "1/384"])
        self.assertEqual(("""
Orig. box coord. = 0.0,50.0,5.0,52.5
Adj. box coord.  = 0.0,49.33333333333333,5.333333333333333,53.33333333333333
Orig. box WKT    = POLYGON ((0.0 50.0, 5.0 50.0, 5.0 52.5, 0.0 52.5, 0.0 50.0))
Adj. box WKT     = POLYGON ((0.0 49.33333333333333, 5.333333333333333 49.33333333333333, 5.333333333333333 53.33333333333333, 0.0 53.33333333333333, 0.0 49.33333333333333))
Combined WKT     = MULTIPOLYGON (((0.0 50.0, 5.0 50.0, 5.0 52.5, 0.0 52.5, 0.0 50.0)), ((0.0 49.33333333333333, 5.333333333333333 49.33333333333333, 5.333333333333333 53.33333333333333, 0.0 53.33333333333333, 0.0 49.33333333333333)))
Box grid size    = 2048 x 1536 cells
Graticule dist.  = 4/3 degrees
Tile size        = 1 cells
Granularity      = 135 cells
Level            = 9
Res. at level 0  = 4/3 degrees
Resolution       = 1/384 degrees
                 = 289.89 meters
"""),
                         result.output)
        self.assertEqual(0, result.exit_code)
