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

import json
import os
import os.path
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from test.mixins import AlmostEqualDeepMixin
from test.webapi.helpers import get_api_ctx
from xcube.webapi.timeseries.context import TimeSeriesContext
from xcube.webapi.timeseries.controllers import collect_timeseries_result
from xcube.webapi.timeseries.controllers import get_time_series


def get_timeseries_ctx(server_config=None) -> TimeSeriesContext:
    return get_api_ctx("timeseries", TimeSeriesContext, server_config)


class TimeSeriesControllerTest(unittest.TestCase, AlmostEqualDeepMixin):

    def setUp(self) -> None:
        self.maxDiff = None

    def test_get_time_series_for_point(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Point",
                                             coordinates=[2.1, 51.4]),
                                        start_date=np.datetime64(
                                            '2017-01-15'),
                                        end_date=np.datetime64('2017-01-29'))
        expected_result = [
            {'mean': 3.534773588180542, 'time': '2017-01-16T10:09:22Z'},
            {'mean': None, 'time': '2017-01-25T09:35:51Z'},
            {'mean': None, 'time': '2017-01-26T10:50:17Z'},
            {'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_with_tolerance(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(
            ctx, 'demo', 'conc_tsm',
            dict(type="Point",
                 coordinates=[2.1, 51.4]),
            start_date=np.datetime64(
                '2017-01-16T10:10:00Z'
            ),
            end_date=np.datetime64(
                '2017-01-28T09:55:00Z'
            ),
            tolerance=0
        )
        # Outer values missing!
        # see https://github.com/dcs4cop/xcube/issues/860
        expected_result = [
            {'mean': None, 'time': '2017-01-25T09:35:51Z'},
            {'mean': None, 'time': '2017-01-26T10:50:17Z'},
        ]
        self.assertAlmostEqualDeep(expected_result, actual_result)

        actual_result = get_time_series(
            ctx, 'demo', 'conc_tsm',
            dict(type="Point",
                 coordinates=[2.1, 51.4]),
            start_date=np.datetime64(
                '2017-01-16T10:00:00Z'
            ),
            end_date=np.datetime64(
                '2017-01-28T09:55:00Z'
            ),
            tolerance=5 * 60   # 5 minutes
        )
        expected_result = [
            {'mean': 3.534773588180542, 'time': '2017-01-16T10:09:22Z'},
            {'mean': None, 'time': '2017-01-25T09:35:51Z'},
            {'mean': None, 'time': '2017-01-26T10:50:17Z'},
            {'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_point_out_of_bounds(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Point",
                                             coordinates=[-150.0, -30.0]))
        expected_result = []
        self.assertEqual(expected_result, actual_result)

    def test_get_time_series_for_point_one_valid(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Point",
                                             coordinates=[2.1, 51.4]),
                                        agg_methods=['mean', 'count'],
                                        start_date=np.datetime64(
                                            '2017-01-15'),
                                        end_date=np.datetime64('2017-01-29'),
                                        max_valids=1)
        expected_result = [
            {'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_point_only_valids(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Point",
                                             coordinates=[2.1, 51.4]),
                                        start_date=np.datetime64(
                                            '2017-01-15'),
                                        end_date=np.datetime64('2017-01-29'),
                                        max_valids=-1)
        expected_result = [
            {'mean': 3.534773588180542, 'time': '2017-01-16T10:09:22Z'},
            {'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_polygon(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Polygon", coordinates=[[
                                            [1., 51.], [2., 51.], [2., 52.],
                                            [1., 52.], [1., 51.]
                                        ]]),
                                        agg_methods=['mean', 'count'])
        expected_result = [{'count': 122392,
                            'count_tot': 159600,
                            'mean': 56.12519223634024,
                            'time': '2017-01-16T10:09:22Z'},
                           {'count': 0,
                            'count_tot': 159600,
                            'mean': None,
                            'time': '2017-01-25T09:35:51Z'},
                           {'count': 0,
                            'count_tot': 159600,
                            'mean': None,
                            'time': '2017-01-26T10:50:17Z'},
                           {'count': 132066,
                            'count_tot': 159600,
                            'mean': 49.70755256053988,
                            'time': '2017-01-28T09:58:11Z'},
                           {'count': 0,
                            'count_tot': 159600,
                            'mean': None,
                            'time': '2017-01-30T10:46:34Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_polygon_with_all_agg_methods(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Polygon", coordinates=[[
                                            [1., 51.], [2., 51.], [2., 52.],
                                            [1., 52.], [1., 51.]
                                        ]]),
                                        agg_methods=['mean', 'median', 'std',
                                                     'min', 'max', 'count'])

        expected_result = [{'count': 122392,
                            'count_tot': 159600,
                            'max': 166.57278442382812,
                            'mean': 56.12519223634024,
                            'median': 48.009796142578125,
                            'min': 0.02251400426030159,
                            'std': 40.78859862094861,
                            'time': '2017-01-16T10:09:22Z'},
                           {'count': 0,
                            'count_tot': 159600,
                            'max': None,
                            'mean': None,
                            'median': None,
                            'min': None,
                            'std': None,
                            'time': '2017-01-25T09:35:51Z'},
                           {'count': 0,
                            'count_tot': 159600,
                            'max': None,
                            'mean': None,
                            'median': None,
                            'min': None,
                            'std': None,
                            'time': '2017-01-26T10:50:17Z'},
                           {'count': 132066,
                            'count_tot': 159600,
                            'max': 158.7908477783203,
                            'mean': 49.70755256053988,
                            'median': 39.326446533203125,
                            'min': 0.02607600949704647,
                            'std': 34.98868194514787,
                            'time': '2017-01-28T09:58:11Z'},
                           {'count': 0,
                            'count_tot': 159600,
                            'max': None,
                            'mean': None,
                            'median': None,
                            'min': None,
                            'std': None,
                            'time': '2017-01-30T10:46:34Z'}]

        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_polygon_one_valid(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Polygon", coordinates=[[
                                            [1., 51.], [2., 51.], [2., 52.],
                                            [1., 52.], [1., 51.]
                                        ]]),
                                        agg_methods=['mean', 'count'],
                                        max_valids=1)
        expected_result = [{'count': 132066,
                            'count_tot': 159600,
                            'mean': 49.70755256053988,
                            'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_polygon_only_valids(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Polygon", coordinates=[[
                                            [1., 51.], [2., 51.], [2., 52.],
                                            [1., 52.], [1., 51.]
                                        ]]),
                                        agg_methods=['mean', 'count'],
                                        max_valids=-1)
        expected_result = [{'count': 122392,
                            'count_tot': 159600,
                            'mean': 56.12519223634024,
                            'time': '2017-01-16T10:09:22Z'},
                           {'count': 132066,
                            'count_tot': 159600,
                            'mean': 49.70755256053988,
                            'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_point_collection(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx,
                                        'demo', 'conc_tsm',
                                        dict(type="GeometryCollection",
                                             geometries=[
                                                 dict(type="Point",
                                                      coordinates=[2.1,
                                                                   51.4])]),
                                        agg_methods=['mean', 'count'],
                                        start_date=np.datetime64(
                                            '2017-01-15'),
                                        end_date=np.datetime64('2017-01-29'))
        expected_result = [
            [{'mean': 3.534773588180542, 'time': '2017-01-16T10:09:22Z'},
             {'mean': None, 'time': '2017-01-25T09:35:51Z'},
             {'mean': None, 'time': '2017-01-26T10:50:17Z'},
             {'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_polygon_collection(self):
        ctx = get_timeseries_ctx()
        actual_result = get_time_series(ctx,
                                        'demo', 'conc_tsm',
                                        dict(type="GeometryCollection",
                                             geometries=[dict(type="Polygon",
                                                              coordinates=[[
                                                                  [1., 51.],
                                                                  [2., 51.],
                                                                  [2., 52.],
                                                                  [1., 52.],
                                                                  [1., 51.]
                                                              ]])]),
                                        agg_methods=['mean', 'count'])
        expected_result = [[{'count': 122392,
                             'count_tot': 159600,
                             'mean': 56.12519223634024,
                             'time': '2017-01-16T10:09:22Z'},
                            {'count': 0,
                             'count_tot': 159600,
                             'mean': None,
                             'time': '2017-01-25T09:35:51Z'},
                            {'count': 0,
                             'count_tot': 159600,
                             'mean': None,
                             'time': '2017-01-26T10:50:17Z'},
                            {'count': 132066,
                             'count_tot': 159600,
                             'mean': 49.70755256053988,
                             'time': '2017-01-28T09:58:11Z'},
                            {'count': 0,
                             'count_tot': 159600,
                             'mean': None,
                             'time': '2017-01-30T10:46:34Z'}]]

        self.assertAlmostEqualDeep(expected_result, actual_result)


class CollectTimeSeriesResultTest(unittest.TestCase, AlmostEqualDeepMixin):
    expected_result = [
        {'a': True, 'b': 32, 'c': 0.4, 'time': '2010-04-05T00:00:00Z'},
        {'a': False, 'b': 33, 'c': 0.2, 'time': '2010-04-06T00:00:00Z'},
        {'a': False, 'b': 35, 'c': None, 'time': '2010-04-07T00:00:00Z'},
        {'a': True, 'b': 34, 'c': 0.7, 'time': '2010-04-08T00:00:00Z'}
    ]

    expected_json = (
        '[{"a": true, "b": 32, "c": 0.4, "time": "2010-04-05T00:00:00Z"},'
        ' {"a": false, "b": 33, "c": 0.2, "time": "2010-04-06T00:00:00Z"},'
        ' {"a": false, "b": 35, "c": null, "time": "2010-04-07T00:00:00Z"},'
        ' {"a": true, "b": 34, "c": 0.7, "time": "2010-04-08T00:00:00Z"}]'
    )

    def test_all_types_converted_correctly(self):
        self.maxDiff = None

        time_series_ds = xr.Dataset(
            data_vars=dict(time=xr.DataArray(
                pd.date_range(start='2010-04-05', periods=4, freq='1D'),
                dims='time'),
                a=xr.DataArray([True, False, False, True],
                               dims='time'),
                b=xr.DataArray([32, 33, 35, 34], dims='time'),
                c=xr.DataArray([0.4, 0.2, np.nan, 0.7],
                               dims='time'))
        )
        result = collect_timeseries_result(
            time_series_ds,
            {'a': 'a', 'b': 'b', 'c': 'c'},
            max_valids=None
        )

        self.assertEqual(self.expected_result, result)
        self.assertEqual(self.expected_json, json.dumps(result))

    def test_num_valids(self):
        self.maxDiff = None

        time_series_ds = xr.Dataset(
            data_vars=dict(time=xr.DataArray(
                pd.date_range(start='2010-04-05', periods=5, freq='1D'),
                dims='time'),
                a=xr.DataArray([np.nan, 5, np.nan, 2, 3],
                               dims='time'),
                b=xr.DataArray([np.nan, 33, np.nan, np.nan, 23],
                               dims='time'),
                c=xr.DataArray([np.nan, 0.2, np.nan, 0.7, np.nan],
                               dims='time')))

        result = collect_timeseries_result(time_series_ds,
                                           {'a': 'a', 'b': 'b', 'c': 'c'},
                                           max_valids=-1)
        self.assertEqual(
            [{'a': 5.0, 'b': 33.0, 'c': 0.2, 'time': '2010-04-06T00:00:00Z'},
             {'a': 2.0, 'b': None, 'c': 0.7, 'time': '2010-04-08T00:00:00Z'},
             {'a': 3.0, 'b': 23.0, 'c': None,
              'time': '2010-04-09T00:00:00Z'}],
            result)

        result = collect_timeseries_result(time_series_ds,
                                           {'a': 'a', 'b': 'b', 'c': 'c'},
                                           max_valids=2)
        self.assertEqual(
            [{'a': 2.0, 'b': None, 'c': 0.7, 'time': '2010-04-08T00:00:00Z'},
             {'a': 3.0, 'b': 23.0, 'c': None,
              'time': '2010-04-09T00:00:00Z'}],
            result)

    def test_count(self):
        self.maxDiff = None

        time_series_ds = xr.Dataset(
            data_vars=dict(time=xr.DataArray(
                pd.date_range(start='2010-04-05', periods=3, freq='1D'),
                dims='time'),
                c=xr.DataArray([23, 78, 74], dims='time')),
            attrs=dict(max_number_of_observations=82))

        result = collect_timeseries_result(time_series_ds, {'count': 'c'},
                                           max_valids=-1)
        self.assertEqual(
            [{'count': 23, 'count_tot': 82, 'time': '2010-04-05T00:00:00Z'},
             {'count': 78, 'count_tot': 82, 'time': '2010-04-06T00:00:00Z'},
             {'count': 74, 'count_tot': 82, 'time': '2010-04-07T00:00:00Z'}],
            result)


@unittest.skipUnless(os.environ.get('XCUBE_TS_PERF_TEST') == '1',
                     'XCUBE_TS_PERF_TEST is not 1')
class TsPerfTest(unittest.TestCase):

    def test_point_ts_perf(self):
        test_cube = 'ts_test.zarr'

        if not os.path.isdir(test_cube):
            from xcube.core.new import new_cube
            cube = new_cube(time_periods=2000,
                            variables=dict(analysed_sst=280.4))
            cube = cube.chunk(dict(time=1, lon=90, lat=90))
            cube.to_zarr(test_cube)

        ctx = get_timeseries_ctx(dict(
            base_dir='.',
            Datasets=[
                dict(
                    Identifier='ts_test',
                    FileSystem='file',
                    Path=test_cube,
                    Format='zarr'
                )
            ]
        ))

        num_repeats = 5
        import random
        import time
        time_sum = 0.0
        for i in range(num_repeats):
            lon = -180 + 360 * random.random()
            lat = -90 + 180 * random.random()

            t1 = time.perf_counter()
            result = get_time_series(ctx, 'ts_test', 'analysed_sst',
                                     dict(type='Point',
                                          coordinates=[lon, lat]))
            t2 = time.perf_counter()

            self.assertIsInstance(result, list)
            self.assertEqual(2000, len(result))

            time_delta = t2 - t1
            time_sum += time_delta
            print(f'test {i + 1} took {time_delta} seconds')

        print(f'all tests took {time_sum / num_repeats} seconds in average')
