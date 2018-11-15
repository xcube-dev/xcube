import fractions
import math
import unittest
from typing import List

import click.testing

from xcube.grid.cli import cli, factor_out_two, get_adjusted_box, get_levels, find_close_resolutions, meters_to_degrees, \
    degrees_to_meters


def find_stuff(target_res, delta_res, min_level=0):
    if target_res <= 0.0:
        raise ValueError('illegal target_res')
    if delta_res < 0.0 or delta_res >= target_res:
        raise ValueError('illegal delta_res')
    if min_level < 0.0:
        raise ValueError('illegal min_level')
    res_1 = target_res - delta_res
    res_2 = target_res + delta_res
    h_1 = 180 / res_1
    h_2 = 180 / res_2
    if h_2 < h_1:
        h_1, h_2 = h_2, h_1
    h_1 = int(math.floor(h_1))
    h_2 = int(math.ceil(h_2))
    results = []
    for h in range(h_1, h_2 + 1):
        res = fractions.Fraction(180, h)
        res_f = float(res)
        delta = res_f - target_res
        if abs(delta) <= delta_res:
            cov = res * h
            if int(cov) == 180 and int(cov) == float(cov):
                h0, level = factor_out_two(h)
                if level >= min_level:
                    delta_p = 100 * delta / target_res
                    delta_p = round(10 * delta_p) / 10
                    res_m = degrees_to_meters(res_f)
                    results.append((delta_p, res.numerator, res.denominator, res_f, res_m, h, h0, level))
    return sorted(results, key=lambda item: abs(item[0]))


import pprint


class GridToolsTest(unittest.TestCase):
    def test_find_stuff(self):
        target_res = 0.0833333
        delta_res = 0.025 * target_res
        results = find_stuff(target_res, delta_res, min_level=3)
        pprint.pprint(results)
        print()

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
        results = get_levels(3, 4)
        self.assertEqual(5, len(results))
        self.assertEqual([0, 1, 2, 3, 4],
                         [row[0] for row in results])
        self.assertEqual([540, 1080, 2160, 4320, 8640],
                         [row[1] for row in results])
        self.assertEqual([3, 6, 12, 24, 48],
                         [row[2] for row in results])

    def test_find_close_resolutions(self):
        target_res = meters_to_degrees(300)
        results = find_close_resolutions(target_res, target_res * 5 / 100)
        self.assertEqual(9, len(results))
        self.assertEqual([540, 4140, 8100, 8460, 16020, 16380, 16740, 17100, 17460],
                         [row[0] for row in results])
        self.assertEqual([7, 4, 3, 3, 2, 2, 2, 2, 2],
                         [row[1] for row in results])
        self.assertEqual([69120, 66240, 64800, 67680, 64080, 65520, 66960, 68400, 69840],
                         [row[2] for row in results])
        self.assertEqual([384, 368, 360, 376, 356, 364, 372, 380, 388],
                         [row[3] for row in results])


class GridCliTest(unittest.TestCase):

    @classmethod
    def invoke_cli(cls, args: List[str]):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, args)

    def test_levels(self):
        result = self.invoke_cli(["levels", "384"])
        self.assertEqual(("\n"
                          "LEVEL	HEIGHT	INV_RES	RES (deg)	RES (m)\n"
                          "0	540	3	0.3333333333333333	37106.5\n"
                          "1	1080	6	0.16666666666666666	18553.2\n"
                          "2	2160	12	0.08333333333333333	9276.6\n"
                          "3	4320	24	0.041666666666666664	4638.3\n"
                          "4	8640	48	0.020833333333333332	2319.2\n"
                          "5	17280	96	0.010416666666666666	1159.6\n"
                          "6	34560	192	0.005208333333333333	579.8\n"
                          "7	69120	384	0.0026041666666666665	289.9\n"),
                         result.output)
        self.assertEqual(0, result.exit_code)

    def test_res(self):
        result = self.invoke_cli(["res", "300m", "-d", "5%"])
        self.assertEqual(("\n"
                          "TILE	LEVEL	HEIGHT	INV_RES	RES (deg)	RES (m), DELTA_RES (%)\n"
                          "540	7	69120	384	0.0026041666666666665	289.9	-3.4\n"
                          "4140	4	66240	368	0.002717391304347826	302.5	0.8\n"
                          "8100	3	64800	360	0.002777777777777778	309.2	3.1\n"
                          "8460	3	67680	376	0.0026595744680851063	296.1	-1.3\n"
                          "16020	2	64080	356	0.0028089887640449437	312.7	4.2\n"
                          "16380	2	65520	364	0.0027472527472527475	305.8	1.9\n"
                          "16740	2	66960	372	0.002688172043010753	299.2	-0.3\n"
                          "17100	2	68400	380	0.002631578947368421	292.9	-2.4\n"
                          "17460	2	69840	388	0.002577319587628866	286.9	-4.4\n"),
                         result.output)
        self.assertEqual(0, result.exit_code)

    def test_abox(self):
        result = self.invoke_cli(['abox', "0,50,5,52.5", "384"])
        self.assertEqual(("\n"
                          "Orig. box coord. = 0.0,50.0,5.0,52.5\n"
                          "Adj. box coord.  = 0.0,49.21875,5.625,53.4375\n"
                          "Orig. box WKT    = POLYGON ((0.0 50.0, 5.0 50.0, 5.0 52.5, 0.0 52.5, 0.0 50.0))\n"
                          "Adj. box WKT     = POLYGON ((0.0 49.21875, 5.625 49.21875,"
                          " 5.625 53.4375, 0.0 53.4375, 0.0 49.21875))\n"
                          "Grid size  = 2160 x 1620 cells\n"
                          "with\n"
                          "  TILE      = 540\n"
                          "  LEVEL     = 7\n"
                          "  INV_RES   = 384\n"
                          "  RES (deg) = 0.0026041666666666665\n"
                          "  RES (m)   = 289.89450727414993\n"),
                         result.output)
        self.assertEqual(0, result.exit_code)
