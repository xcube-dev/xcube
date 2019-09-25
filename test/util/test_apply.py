import unittest

from xcube.api import new_cube, chunk_dataset, apply


class ApplyTest(unittest.TestCase):

    def test_apply(self):
        cube = new_cube(width=360, height=180, time_periods=6, variables=dict(analysed_sst=275.3, analysis_error=2.1))
        cube = chunk_dataset(cube, dict(time=3, lat=90, lon=90))

        calls = []

        def my_cube_func(analysed_sst, analysis_error, lat, lon, parameters):
            nonlocal calls
            calls.append((analysed_sst, analysis_error, lat, lon, parameters))
            return analysed_sst + analysis_error

        result_var = apply(cube,
                           my_cube_func,
                           ['analysed_sst', 'analysis_error'],
                           ['lat', 'lon'],
                           params=dict(my_param_1=9.3, my_param_2='Helo'))

        self.assertIsNotNone(result_var)
        self.assertEqual(0, len(calls))

        values = result_var.values
        self.assertEqual((6, 180, 360), values.shape)
        self.assertEqual(2 * 2 * 4, len(calls))
        self.assertEqual(275.3 + 2.1, values[0, 0, 0])
        self.assertEqual(275.3 + 2.1, values[-1, -1, -1])
