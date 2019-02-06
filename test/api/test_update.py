import unittest

import numpy as np
import pandas as pd
import xarray as xr

from test.sampledata import create_highroc_dataset
from xcube.api.update import update_var_props, update_global_attrs


class UpdateVariablePropsTest(unittest.TestCase):
    def test_no_change(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = update_var_props(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = update_var_props(ds1, [])
        self.assertIs(ds2, ds1)

    def test_change_all_or_none(self):
        ds1 = create_highroc_dataset()
        ds2 = update_var_props(ds1,
                               [(var_name, {'marker': True}) for var_name in ds1.data_vars])
        self.assertEqual(len(ds1.data_vars), len(ds2.data_vars))
        self.assertTrue(all(['marker' in ds2[n].attrs for n in ds2.variables]))

        with self.assertRaises(KeyError):
            update_var_props(ds1, [('bibo', {'marker': True})])

    def test_change_some(self):
        ds1 = create_highroc_dataset()
        ds2 = update_var_props(ds1,
                               [('conc_chl', {'name': 'chl_c2rcc'}),
                                     ('c2rcc_flags', {'name': 'flags', 'marker': True}),
                                     ('rtoa_10', None)])

        self.assertEqual(len(ds1.data_vars), len(ds2.data_vars))

        self.assertNotIn('conc_chl', ds2.data_vars)
        self.assertNotIn('c2rcc_flags', ds2.data_vars)

        self.assertIn('chl_c2rcc', ds2.data_vars)
        self.assertIn('original_name', ds2.chl_c2rcc.attrs)
        self.assertEqual('conc_chl', ds2.chl_c2rcc.attrs['original_name'])

        self.assertIn('flags', ds2.data_vars)
        self.assertIn('original_name', ds2.flags.attrs)
        self.assertEqual('c2rcc_flags', ds2.flags.attrs['original_name'])
        self.assertIn('marker', ds2.flags.attrs)
        self.assertEqual(True, ds2.flags.attrs['marker'])

        self.assertIn('rtoa_10', ds2.data_vars)

        with self.assertRaises(ValueError) as cm:
            update_var_props(ds1, [('conc_chl', None),
                                   ('c2rcc_flags', None),
                                   ('rtoa_1', {'name': 'refl_toa'}),
                                   ('rtoa_2', {'name': 'refl_toa'}),
                                   ('rtoa_3', {'name': 'refl_toa'})])
        self.assertEqual("variable 'rtoa_2' cannot be renamed into 'refl_toa' because the name is already in use",
                         f'{cm.exception}')


class UpdateGlobalAttributesTest(unittest.TestCase):

    def test_update_global_attributes(self):
        num_lons = 8
        num_lats = 6
        num_times = 5

        lon_min = -20.
        lat_min = 12.

        res = 0.25
        res05 = res / 2
        lon = np.linspace(lon_min + res05, lon_min + num_lons * res - res05, num_lons)
        lat = np.linspace(lat_min + res05, lat_min + num_lats * res - res05, num_lats)
        lon_bnds = np.array([[v - res05, v + res05] for v in lon])
        lat_bnds = np.array([[v - res05, v + res05] for v in lat])
        time = [pd.to_datetime(f'2018-06-0{i}T12:00:00') for i in range(1, num_times + 1)]
        time_bnds = [(pd.to_datetime(f'2018-06-0{i}T00:00:00'),
                      pd.to_datetime(f'2018-06-0{i}T23:00:59')) for i in range(1, num_times + 1)]

        coords = dict(time=(['time'], time),
                      lat=(['lat'], lat),
                      lon=(['lon'], lon))

        coords_with_bnds = dict(time_bnds=(['time', 'bnds'], time_bnds),
                                lat_bnds=(['lat', 'bnds'], lat_bnds),
                                lon_bnds=(['lon', 'bnds'], lon_bnds),
                                **coords)

        output_metadata = dict(history='pipo', license='MIT', Conventions='CF-1.7')

        ds1 = xr.Dataset(coords=coords)
        ds2 = update_global_attrs(ds1, output_metadata=output_metadata)

        self.assertIsNot(ds2, ds1)
        self.assertEqual('CF-1.7', ds2.attrs.get('Conventions'))
        self.assertEqual('MIT', ds2.attrs.get('license'))
        self.assertEqual('pipo', ds2.attrs.get('history'))
        self.assertEqual(-20.0, ds2.attrs.get('geospatial_lon_min'))
        self.assertEqual(-18.0, ds2.attrs.get('geospatial_lon_max'))
        self.assertEqual(0.25, ds2.attrs.get('geospatial_lon_resolution'))
        self.assertEqual('degrees_east', ds2.attrs.get('geospatial_lon_units'))
        self.assertEqual(12.0, ds2.attrs.get('geospatial_lat_min'))
        self.assertEqual(13.5, ds2.attrs.get('geospatial_lat_max'))
        self.assertEqual(0.25, ds2.attrs.get('geospatial_lat_resolution'))
        self.assertEqual('degrees_north', ds2.attrs.get('geospatial_lat_units'))
        self.assertEqual('2018-06-01T00:00:00.000000000', ds2.attrs.get('time_coverage_start'))
        self.assertEqual('2018-06-06T00:00:00.000000000', ds2.attrs.get('time_coverage_end'))
        self.assertIn('date_modified', ds2.attrs)

        ds1 = xr.Dataset(coords=coords_with_bnds)
        ds2 = update_global_attrs(ds1, output_metadata=output_metadata)

        self.assertIsNot(ds2, ds1)
        self.assertEqual('CF-1.7', ds2.attrs.get('Conventions'))
        self.assertEqual('MIT', ds2.attrs.get('license'))
        self.assertEqual('pipo', ds2.attrs.get('history'))
        self.assertEqual(-20.0, ds2.attrs.get('geospatial_lon_min'))
        self.assertEqual(-18.0, ds2.attrs.get('geospatial_lon_max'))
        self.assertEqual(0.25, ds2.attrs.get('geospatial_lon_resolution'))
        self.assertEqual('degrees_east', ds2.attrs.get('geospatial_lon_units'))
        self.assertEqual(12.0, ds2.attrs.get('geospatial_lat_min'))
        self.assertEqual(13.5, ds2.attrs.get('geospatial_lat_max'))
        self.assertEqual(0.25, ds2.attrs.get('geospatial_lat_resolution'))
        self.assertEqual('degrees_north', ds2.attrs.get('geospatial_lat_units'))
        self.assertEqual('2018-06-01T00:00:00.000000000', ds2.attrs.get('time_coverage_start'))
        self.assertEqual('2018-06-05T23:00:59.000000000', ds2.attrs.get('time_coverage_end'))
        self.assertIn('date_modified', ds2.attrs)
