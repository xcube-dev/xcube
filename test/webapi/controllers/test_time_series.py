import unittest

import numpy as np
import xarray as xr

from xcube.webapi.controllers.time_series import get_time_series_info, get_time_series_for_point, \
    get_time_series_for_geometry, get_time_series_for_geometry_collection, _find_ancillary_var_name, \
    get_dataset_and_variable
from xcube.webapi.errors import ServiceResourceNotFoundError
from ..helpers import new_test_service_context


class TimeSeriesControllerTest(unittest.TestCase):

    def test_get_time_series_for_point_invalid_lat_and_lon(self):
        ctx = new_test_service_context()
        time_series = get_time_series_for_point(ctx, 'demo', 'conc_tsm',
                                                lon=-150.0, lat=-30.0)
        expected_dict = {'results': []}
        self.assertEqual(expected_dict, time_series)

    def test_get_time_series_for_point(self):
        ctx = new_test_service_context()
        time_series = get_time_series_for_point(ctx, 'demo', 'conc_tsm',
                                                lon=2.1, lat=51.4,
                                                start_date=np.datetime64('2017-01-15'),
                                                end_date=np.datetime64('2017-01-29'))
        expected_dict = {'results': [{'date': '2017-01-16T10:09:22Z',
                                      'result': {'average': 3.534773588180542,
                                                 'totalCount': 1,
                                                 'validCount': 1}},
                                     {'date': '2017-01-25T09:35:51Z',
                                      'result': {'average': None, 'totalCount': 1, 'validCount': 0}},
                                     {'date': '2017-01-26T10:50:17Z',
                                      'result': {'average': None, 'totalCount': 1, 'validCount': 0}},
                                     {'date': '2017-01-28T09:58:11Z',
                                      'result': {'average': 20.12085723876953,
                                                 'totalCount': 1,
                                                 'validCount': 1}}]}
        self.assertEqual(expected_dict, time_series)

    def test_get_time_series_for_point_with_stdev(self):
        ctx = new_test_service_context()
        time_series = get_time_series_for_point(ctx, 'demo-1w', 'conc_tsm',
                                                lon=2.1, lat=51.4,
                                                start_date=np.datetime64('2017-01-15'),
                                                end_date=np.datetime64('2017-01-29'))
        expected_dict = {'results': [{'date': '2017-01-22T00:00:00Z',
                                      'result': {'average': 3.534773588180542,
                                                 'stdev': 0.0,
                                                 'totalCount': 1,
                                                 'validCount': 1}},
                                     {'date': '2017-01-29T00:00:00Z',
                                      'result': {'average': 20.12085723876953,
                                                 'stdev': 0.0,
                                                 'totalCount': 1,
                                                 'validCount': 1}}]}
        self.assertEqual(expected_dict, time_series)

    def test_get_time_series_for_geometry(self):
        ctx = new_test_service_context()
        time_series = get_time_series_for_geometry(ctx, 'demo', 'conc_tsm',
                                                   dict(type="Point", coordinates=[2.1, 51.4]),
                                                   start_date=np.datetime64('2017-01-15'),
                                                   end_date=np.datetime64('2017-01-29'))
        expected_dict = {'results': [{'date': '2017-01-16T10:09:22Z',
                                      'result': {'average': 3.534773588180542,
                                                 'totalCount': 1,
                                                 'validCount': 1}},
                                     {'date': '2017-01-25T09:35:51Z',
                                      'result': {'average': None, 'totalCount': 1, 'validCount': 0}},
                                     {'date': '2017-01-26T10:50:17Z',
                                      'result': {'average': None, 'totalCount': 1, 'validCount': 0}},
                                     {'date': '2017-01-28T09:58:11Z',
                                      'result': {'average': 20.12085723876953,
                                                 'totalCount': 1,
                                                 'validCount': 1}}]}
        self.assertEqual(expected_dict, time_series)

        time_series = get_time_series_for_geometry(ctx, 'demo', 'conc_tsm',
                                                   dict(type="Polygon", coordinates=[[
                                                       [1., 51.], [2., 51.], [2., 52.], [1., 52.], [1., 51.]
                                                   ]]))
        expected_dict = {'results': [
            {'result': {'totalCount': 160801, 'validCount': 123540, 'average': 56.04547741839104},
             'date': '2017-01-16T10:09:22Z'},
            {'result': {'totalCount': 160801, 'validCount': 0, 'average': None},
             'date': '2017-01-25T09:35:51Z'},
            {'result': {'totalCount': 160801, 'validCount': 0, 'average': None},
             'date': '2017-01-26T10:50:17Z'},
            {'result': {'totalCount': 160801, 'validCount': 133267, 'average': 49.59349042604672},
             'date': '2017-01-28T09:58:11Z'},
            {'result': {'totalCount': 160801, 'validCount': 0, 'average': None},
             'date': '2017-01-30T10:46:34Z'}]}

        self.assertEqual(expected_dict, time_series)

    def test_get_time_series_for_geometry_with_stdev(self):
        ctx = new_test_service_context()
        time_series = get_time_series_for_geometry(ctx, 'demo-1w', 'conc_tsm',
                                                   dict(type="Point", coordinates=[2.1, 51.4]),
                                                   start_date=np.datetime64('2017-01-15'),
                                                   end_date=np.datetime64('2017-01-29'))
        expected_dict = {'results': [{'date': '2017-01-22T00:00:00Z',
                                      'result': {'average': 3.534773588180542,
                                                 'stdev': 0.0,
                                                 'totalCount': 1,
                                                 'validCount': 1}},
                                     {'date': '2017-01-29T00:00:00Z',
                                      'result': {'average': 20.12085723876953,
                                                 'stdev': 0.0,
                                                 'totalCount': 1,
                                                 'validCount': 1}}]}
        self.assertEqual(expected_dict, time_series)

        time_series = get_time_series_for_geometry(ctx, 'demo-1w', 'conc_tsm',
                                                   dict(type="Polygon", coordinates=[[
                                                       [1., 51.], [2., 51.], [2., 52.], [1., 52.], [1., 51.]
                                                   ]]))
        expected_dict = {'results': [{'date': '2017-01-22T00:00:00Z',
                                      'result': {'average': 56.04547741839104,
                                                 'stdev': 0.0,
                                                 'totalCount': 160801,
                                                 'validCount': 123540}},
                                     {'date': '2017-01-29T00:00:00Z',
                                      'result': {'average': 49.59349042604672,
                                                 'stdev': 0.0,
                                                 'totalCount': 160801,
                                                 'validCount': 133267}},
                                     {'date': '2017-02-05T00:00:00Z',
                                      'result': {'average': None,
                                                 'stdev': None,
                                                 'totalCount': 160801,
                                                 'validCount': 0}}]}

        self.assertEqual(expected_dict, time_series)

    def test_get_time_series_for_geometries(self):
        ctx = new_test_service_context()
        time_series = get_time_series_for_geometry_collection(ctx,
                                                              'demo', 'conc_tsm',
                                                              dict(type="GeometryCollection",
                                                                   geometries=[
                                                                       dict(type="Point", coordinates=[2.1, 51.4])]),
                                                              start_date=np.datetime64('2017-01-15'),
                                                              end_date=np.datetime64('2017-01-29'))
        expected_dict = {'results': [[{'date': '2017-01-16T10:09:22Z',
                                       'result': {'average': 3.534773588180542,
                                                  'totalCount': 1,
                                                  'validCount': 1}},
                                      {'date': '2017-01-25T09:35:51Z',
                                       'result': {'average': None, 'totalCount': 1, 'validCount': 0}},
                                      {'date': '2017-01-26T10:50:17Z',
                                       'result': {'average': None, 'totalCount': 1, 'validCount': 0}},
                                      {'date': '2017-01-28T09:58:11Z',
                                       'result': {'average': 20.12085723876953,
                                                  'totalCount': 1,
                                                  'validCount': 1}}]]}
        self.assertEqual(expected_dict, time_series)

        time_series = get_time_series_for_geometry_collection(ctx,
                                                              'demo', 'conc_tsm',
                                                              dict(type="GeometryCollection",
                                                                   geometries=[dict(type="Polygon", coordinates=[[
                                                                       [1., 51.], [2., 51.], [2., 52.], [1., 52.],
                                                                       [1., 51.]
                                                                   ]])]))
        expected_dict = {'results': [[
            {'result': {'totalCount': 160801, 'validCount': 123540, 'average': 56.04547741839104},
             'date': '2017-01-16T10:09:22Z'},
            {'result': {'totalCount': 160801, 'validCount': 0, 'average': None},
             'date': '2017-01-25T09:35:51Z'},
            {'result': {'totalCount': 160801, 'validCount': 0, 'average': None},
             'date': '2017-01-26T10:50:17Z'},
            {'result': {'totalCount': 160801, 'validCount': 133267, 'average': 49.59349042604672},
             'date': '2017-01-28T09:58:11Z'},
            {'result': {'totalCount': 160801, 'validCount': 0, 'average': None},
             'date': '2017-01-30T10:46:34Z'}]]}

        self.assertEqual(expected_dict, time_series)

    def test_find_ancillary_var_name(self):

        # Find using attribute ancillary_variables and prefix "<name> standard_error"
        ds = xr.Dataset(data_vars=dict(
            analysed_sst=xr.DataArray(np.random.rand(8, 100, 200),
                                      dims=("time", "lat", "lon"),
                                      attrs=dict(standard_name='sea_water_temperature',
                                                 ancillary_variables="analysis_error mask")),
            analysis_error=xr.DataArray(0.01 * np.random.rand(8, 100, 200),
                                        dims=("time", "lat", "lon"),
                                        attrs=dict(standard_name='sea_water_temperature standard_error')),
        ))
        self.assertEqual(('analysis_error', 'error'), _find_ancillary_var_name(ds, ds.analysed_sst))

        # Find using standard_name prefix "<name> standard_error"
        ds = xr.Dataset(data_vars=dict(
            analysed_sst=xr.DataArray(np.random.rand(8, 100, 200),
                                      dims=("time", "lat", "lon"),
                                      attrs=dict(standard_name='sea_water_temperature')),
            analysis_error=xr.DataArray(0.01 * np.random.rand(8, 100, 200),
                                        dims=("time", "lat", "lon"),
                                        attrs=dict(standard_name='sea_water_temperature standard_error')),
        ))
        self.assertEqual(('analysis_error', 'error'), _find_ancillary_var_name(ds, ds.analysed_sst))

        # Find via prefix <name>_stdev
        ds = xr.Dataset(data_vars=dict(
            analysed_sst=xr.DataArray(np.random.rand(8, 100, 200),
                                      dims=("time", "lat", "lon")),
            analysed_sst_stdev=xr.DataArray(0.01 * np.random.rand(8, 100, 200),
                                            dims=("time", "lat", "lon")),
        ))
        self.assertEqual(('analysed_sst_stdev', 'stdev'), _find_ancillary_var_name(ds, ds.analysed_sst))

        # Don't find anything
        ds = xr.Dataset(data_vars=dict(
            analysed_sst=xr.DataArray(np.random.rand(8, 100, 200),
                                      dims=("time", "lat", "lon")),
            analysis_error=xr.DataArray(0.01 * np.random.rand(8, 100, 200),
                                        dims=("time", "lat", "lon")),
        ))
        self.assertEqual((None, None), _find_ancillary_var_name(ds, ds.analysed_sst))

    def test_get_time_series_info(self):
        self.maxDiff = None

        ctx = new_test_service_context()
        info = get_time_series_info(ctx)

        expected_dict = self._get_expected_info_dict()
        self.assertEqual(expected_dict, info)

    def test_get_dataset_and_variable(self):
        ctx = new_test_service_context()
        ds, var = get_dataset_and_variable(ctx, 'demo', 'conc_tsm')
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIsInstance(var, xr.DataArray)

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            get_dataset_and_variable(ctx, 'demox', 'conc_ys')
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Dataset "demox" not found', cm.exception.reason)

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            get_dataset_and_variable(ctx, 'demo', 'conc_ys')
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Variable "conc_ys" not found in dataset "demo"', cm.exception.reason)

    @staticmethod
    def _get_expected_info_dict():
        expected_dict = {'layers': []}
        bounds = {'xmin': 0.0, 'ymin': 50.0, 'xmax': 5.0, 'ymax': 52.5}
        demo_times = ['2017-01-16T10:09:22Z', '2017-01-25T09:35:51Z', '2017-01-26T10:50:17Z',
                      '2017-01-28T09:58:11Z', '2017-01-30T10:46:34Z']
        demo_variables = ['quality_flags', 'kd489', 'conc_tsm', 'conc_chl', 'c2rcc_flags']
        for demo_variable in demo_variables:
            dict_variable = {'name': f'demo.{demo_variable}', 'dates': demo_times, 'bounds': bounds}
            expected_dict['layers'].append(dict_variable)
        demo1w_times = ['2017-01-22T00:00:00Z', '2017-01-29T00:00:00Z', '2017-02-05T00:00:00Z']
        for demo_variable in demo_variables:
            dict_variable = {'name': f'demo-1w.{demo_variable}', 'dates': demo1w_times, 'bounds': bounds}
            expected_dict['layers'].append(dict_variable)
        for demo_variable in demo_variables:
            dict_variable = {'name': f'demo-1w.{demo_variable}_stdev', 'dates': demo1w_times, 'bounds': bounds}
            expected_dict['layers'].append(dict_variable)
        return expected_dict
