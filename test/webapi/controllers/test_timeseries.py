import unittest

import numpy as np

from test.mixins import AlmostEqualDeepMixin
from test.webapi.helpers import new_test_service_context
from xcube.webapi.controllers.timeseries import get_time_series


class TimeSeriesControllerTest(unittest.TestCase, AlmostEqualDeepMixin):

    def setUp(self) -> None:
        self.maxDiff = None

    def test_get_time_series_for_point(self):
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Point", coordinates=[2.1, 51.4]),
                                        start_date=np.datetime64('2017-01-15'),
                                        end_date=np.datetime64('2017-01-29'))
        expected_result = [{'mean': 3.534773588180542, 'time': '2017-01-16T10:09:22Z'},
                           {'mean': None, 'time': '2017-01-25T09:35:51Z'},
                           {'mean': None, 'time': '2017-01-26T10:50:17Z'},
                           {'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_point_out_of_bounds(self):
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Point", coordinates=[-150.0, -30.0]))
        expected_result = []
        self.assertEqual(expected_result, actual_result)

    def test_get_time_series_for_point_one_valid(self):
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Point", coordinates=[2.1, 51.4]),
                                        agg_methods=['mean', 'count'],
                                        start_date=np.datetime64('2017-01-15'),
                                        end_date=np.datetime64('2017-01-29'),
                                        max_valids=1)
        expected_result = [{'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_point_only_valids(self):
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Point", coordinates=[2.1, 51.4]),
                                        start_date=np.datetime64('2017-01-15'),
                                        end_date=np.datetime64('2017-01-29'),
                                        max_valids=-1)
        expected_result = [{'mean': 3.534773588180542, 'time': '2017-01-16T10:09:22Z'},
                           {'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_polygon(self):
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Polygon", coordinates=[[
                                            [1., 51.], [2., 51.], [2., 52.], [1., 52.], [1., 51.]
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
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Polygon", coordinates=[[
                                            [1., 51.], [2., 51.], [2., 52.], [1., 52.], [1., 51.]
                                        ]]),
                                        agg_methods=['mean', 'median', 'std', 'min', 'max', 'count'])

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
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Polygon", coordinates=[[
                                            [1., 51.], [2., 51.], [2., 52.], [1., 52.], [1., 51.]
                                        ]]),
                                        agg_methods=['mean', 'count'],
                                        max_valids=1)
        expected_result = [{'count': 132066,
                            'count_tot': 159600,
                            'mean': 49.70755256053988,
                            'time': '2017-01-28T09:58:11Z'}]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_polygon_only_valids(self):
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx, 'demo', 'conc_tsm',
                                        dict(type="Polygon", coordinates=[[
                                            [1., 51.], [2., 51.], [2., 52.], [1., 52.], [1., 51.]
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
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx,
                                        'demo', 'conc_tsm',
                                        dict(type="GeometryCollection",
                                             geometries=[
                                                 dict(type="Point", coordinates=[2.1, 51.4])]),
                                        agg_methods=['mean', 'count'],
                                        start_date=np.datetime64('2017-01-15'),
                                        end_date=np.datetime64('2017-01-29'))
        expected_result = [[{'mean': 3.534773588180542, 'time': '2017-01-16T10:09:22Z'},
                            {'mean': None, 'time': '2017-01-25T09:35:51Z'},
                            {'mean': None, 'time': '2017-01-26T10:50:17Z'},
                            {'mean': 20.12085723876953, 'time': '2017-01-28T09:58:11Z'}]]
        self.assertAlmostEqualDeep(expected_result, actual_result)

    def test_get_time_series_for_polygon_collection(self):
        ctx = new_test_service_context()
        actual_result = get_time_series(ctx,
                                        'demo', 'conc_tsm',
                                        dict(type="GeometryCollection",
                                             geometries=[dict(type="Polygon", coordinates=[[
                                                 [1., 51.], [2., 51.], [2., 52.], [1., 52.],
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
