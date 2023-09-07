import unittest

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray # this is needed for adding crs to a dataset and used as rio

from test.sampledata import create_highroc_dataset
from xcube.core.update import (
    update_dataset_attrs,
    update_dataset_spatial_attrs,
)
from xcube.core.update import update_dataset_var_attrs


class UpdateVariablePropsTest(unittest.TestCase):
    def test_no_change(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = update_dataset_var_attrs(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = update_dataset_var_attrs(ds1, [])
        self.assertIs(ds2, ds1)

    def test_change_all_or_none(self):
        ds1 = create_highroc_dataset()
        ds2 = update_dataset_var_attrs(ds1,
                                       [(var_name, {'marker': True}) for
                                        var_name in ds1.data_vars])
        self.assertEqual(len(ds1.data_vars), len(ds2.data_vars))
        self.assertTrue(all(['marker' in ds2[n].attrs for n in ds2.variables]))

        with self.assertRaises(KeyError):
            update_dataset_var_attrs(ds1, [('bibo', {'marker': True})])

    def test_change_some(self):
        ds1 = create_highroc_dataset()
        ds2 = update_dataset_var_attrs(ds1,
                                       [('conc_chl', {'name': 'chl_c2rcc'}),
                                        ('c2rcc_flags',
                                         {'name': 'flags', 'marker': True}),
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
            update_dataset_var_attrs(ds1, [('conc_chl', None),
                                           ('c2rcc_flags', None),
                                           ('rtoa_1', {'name': 'refl_toa'}),
                                           ('rtoa_2', {'name': 'refl_toa'}),
                                           ('rtoa_3', {'name': 'refl_toa'})])
        self.assertEqual(
            "variable 'rtoa_2' cannot be renamed into 'refl_toa' because the name is already in use",
            f'{cm.exception}'
        )


class UpdateGlobalAttributesTest(unittest.TestCase):

    @staticmethod
    def _create_coords():
        num_lons = 8
        num_lats = 6
        num_times = 5

        lon_min = -20.
        lat_min = 12.

        res = 0.25
        res05 = res / 2
        lon = np.linspace(lon_min + res05, lon_min + num_lons * res - res05,
                          num_lons)
        lat = np.linspace(lat_min + res05, lat_min + num_lats * res - res05,
                          num_lats)
        lon_bnds = np.array([[v - res05, v + res05] for v in lon])
        lat_bnds = np.array([[v - res05, v + res05] for v in lat])
        time = [pd.to_datetime(f'2018-06-0{i}T12:00:00') for i in
                range(1, num_times + 1)]
        time_bnds = [(pd.to_datetime(f'2018-06-0{i}T00:00:00'),
                      pd.to_datetime(f'2018-06-0{i}T23:00:59')) for i in
                     range(1, num_times + 1)]

        coords = dict(time=(['time'], time),
                      lat=(['lat'], lat),
                      lon=(['lon'], lon))

        coords_with_bnds = dict(time_bnds=(['time', 'bnds'], time_bnds),
                                lat_bnds=(['lat', 'bnds'], lat_bnds),
                                lon_bnds=(['lon', 'bnds'], lon_bnds),
                                **coords)

        output_metadata = dict(history='pipo', license='MIT',
                               Conventions='CF-1.7')

        return coords, coords_with_bnds, output_metadata

    def test_update_global_attributes(self):
        coords, coords_with_bnds, output_metadata = self._create_coords()
        ds1 = xr.Dataset(coords=coords)
        ds2 = update_dataset_attrs(ds1, global_attrs=output_metadata)

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
        self.assertEqual('2018-06-01T00:00:00.000000000',
                         ds2.attrs.get('time_coverage_start'))
        self.assertEqual('2018-06-06T00:00:00.000000000',
                         ds2.attrs.get('time_coverage_end'))
        self.assertIn('date_modified', ds2.attrs)

        ds1 = xr.Dataset(coords=coords_with_bnds)
        ds2 = update_dataset_attrs(ds1, global_attrs=output_metadata)

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
        self.assertEqual('2018-06-01T00:00:00.000000000',
                         ds2.attrs.get('time_coverage_start'))
        self.assertEqual('2018-06-05T23:00:59.000000000',
                         ds2.attrs.get('time_coverage_end'))
        self.assertIn('date_modified', ds2.attrs)

    def test_update_global_attributes_crs(self):
        num_x = 8
        num_y = 6
        num_times = 5

        x_min = -20.
        y_min = 12.

        res = 0.25
        res05 = res / 2
        x = np.linspace(x_min + res05, x_min + num_x * res - res05, num_x)
        y = np.linspace(y_min + res05, y_min + num_y * res - res05, num_y)
        x_bnds = np.array([[v - res05, v + res05] for v in x])
        y_bnds = np.array([[v - res05, v + res05] for v in y])
        time = [pd.to_datetime(f'2018-06-0{i}T12:00:00') for i in
                range(1, num_times + 1)]
        time_bnds = [(pd.to_datetime(f'2018-06-0{i}T00:00:00'),
                      pd.to_datetime(f'2018-06-0{i}T23:00:59')) for i in
                     range(1, num_times + 1)]

        coords = dict(time=(['time'], time),
                      y=(['y'], y),
                      x=(['x'], x))

        coords_with_bnds = dict(time_bnds=(['time', 'bnds'], time_bnds),
                                y_bnds=(['y', 'bnds'], y_bnds),
                                x_bnds=(['x', 'bnds'], x_bnds),
                                **coords)

        output_metadata = dict(history='pipo', license='MIT',
                               Conventions='CF-1.7')

        ds1 = xr.Dataset(coords=coords)
        ds1.rio.write_crs("epsg:4326",
                          inplace=True,
                          grid_mapping_name="crs").reset_coords()
        ds2 = update_dataset_attrs(ds1, global_attrs=output_metadata)

        expected_dict = {'history': 'pipo', 'license': 'MIT',
                         'Conventions': 'CF-1.7',
                         'geospatial_lon_units': 'degrees_east',
                         'geospatial_lon_min': -20, 'geospatial_lon_max': -18,
                         'geospatial_lon_resolution': 0.25,
                         'geospatial_lat_units': 'degrees_north',
                         'geospatial_lat_min': 12, 'geospatial_lat_max': 13.5,
                         'geospatial_lat_resolution': 0.25,
                         'geospatial_bounds_crs': 'CRS84',
                         'geospatial_bounds': 'POLYGON((-20 12, -20 13.5, -18 13.5, -18 12, -20 12))',
                         'time_coverage_start': '2018-06-01T00:00:00.000000000',
                         'time_coverage_end': '2018-06-06T00:00:00.000000000'}

        self.assertIsNot(ds2, ds1)
        self.assertIn('date_modified', ds2.attrs)
        ds2.attrs.pop('date_modified')
        self.assertDictEqual(expected_dict,  ds2.attrs)

        ds1 = xr.Dataset(coords=coords_with_bnds)
        ds1.rio.write_crs("epsg:4326",
                          inplace=True,
                          grid_mapping_name="crs").reset_coords()
        ds2 = update_dataset_attrs(ds1, global_attrs=output_metadata)

        expected_dict = {'history': 'pipo', 'license': 'MIT',
                         'Conventions': 'CF-1.7',
                         'geospatial_lon_units': 'degrees_east',
                         'geospatial_lon_min': -20, 'geospatial_lon_max': -18,
                         'geospatial_lon_resolution': 0.25,
                         'geospatial_lat_units': 'degrees_north',
                         'geospatial_lat_min': 12, 'geospatial_lat_max': 13.5,
                         'geospatial_lat_resolution': 0.25,
                         'geospatial_bounds_crs': 'CRS84',
                         'geospatial_bounds': 'POLYGON((-20 12, -20 13.5, -18 13.5, -18 12, -20 12))',
                         'time_coverage_start': '2018-06-01T00:00:00.000000000',
                         'time_coverage_end': '2018-06-05T23:00:59.000000000'}

        self.assertIsNot(ds2, ds1)
        self.assertIn('date_modified', ds2.attrs)
        ds2.attrs.pop('date_modified')
        self.assertDictEqual(expected_dict,  ds2.attrs)

        ds1 = xr.Dataset(coords=coords_with_bnds)
        ds1.rio.write_crs("epsg:4326",
                          inplace=True,
                          grid_mapping_name="crs").reset_coords()
        ds2 = update_dataset_attrs(ds1,
                                   global_attrs=output_metadata,
                                   in_place=False)

        expected_dict = {'history': 'pipo', 'license': 'MIT',
                         'Conventions': 'CF-1.7',
                         'geospatial_lon_units': 'degrees_east',
                         'geospatial_lon_min': -20, 'geospatial_lon_max': -18,
                         'geospatial_lon_resolution': 0.25,
                         'geospatial_lat_units': 'degrees_north',
                         'geospatial_lat_min': 12, 'geospatial_lat_max': 13.5,
                         'geospatial_lat_resolution': 0.25,
                         'geospatial_bounds_crs': 'CRS84',
                         'geospatial_bounds': 'POLYGON((-20 12, -20 13.5, -18 13.5, -18 12, -20 12))',
                         'time_coverage_start': '2018-06-01T00:00:00.000000000',
                         'time_coverage_end': '2018-06-05T23:00:59.000000000'}

        self.assertIsNot(ds2, ds1)
        self.assertIn('date_modified', ds2.attrs)
        ds2.attrs.pop('date_modified')
        self.assertDictEqual(expected_dict,  ds2.attrs)

    def test_update_global_attributes_3031_crs(self):
        num_x = 5
        num_y = 5
        num_times = 5

        x_min = -2602050.
        y_min = -1625550.

        res = 100
        res05 = res / 2
        x = np.linspace(x_min + res05, x_min + num_x * res - res05, num_x)
        y = np.linspace(y_min + res05, y_min + num_y * res - res05, num_y)
        x_bnds = np.array([[v - res05, v + res05] for v in x])
        y_bnds = np.array([[v - res05, v + res05] for v in y])
        time = [pd.to_datetime(f'2018-06-0{i}T12:00:00') for i in
                range(1, num_times + 1)]
        time_bnds = [(pd.to_datetime(f'2018-06-0{i}T00:00:00'),
                      pd.to_datetime(f'2018-06-0{i}T23:00:59')) for i in
                     range(1, num_times + 1)]

        coords = dict(time=(['time'], time),
                      y=(['y'], y),
                      x=(['x'], x))

        coords_with_bnds = dict(time_bnds=(['time', 'bnds'], time_bnds),
                                y_bnds=(['y', 'bnds'], y_bnds),
                                x_bnds=(['x', 'bnds'], x_bnds),
                                **coords)

        output_metadata = dict(history='pipo',
                               license='MIT',
                               Conventions='CF-1.7')

        ds1 = xr.Dataset(coords=coords)
        ds1.rio.write_crs("epsg:3031",
                          inplace=True,
                          grid_mapping_name="crs").reset_coords()
        ds2 = update_dataset_attrs(ds1, global_attrs=output_metadata)

        self.assertIsNot(ds2, ds1)
        self.assertEqual('CF-1.7', ds2.attrs.get('Conventions'))
        self.assertEqual('MIT', ds2.attrs.get('license'))
        self.assertEqual('pipo', ds2.attrs.get('history'))
        self.assertAlmostEqual(-121.99380296455976, ds2.attrs.get('geospatial_lon_min'))
        self.assertAlmostEqual(-121.99083040333379, ds2.attrs.get('geospatial_lon_max'))
        self.assertAlmostEqual(0.0005945389425789926, ds2.attrs.get('geospatial_lon_resolution'))
        self.assertEqual('degrees_east', ds2.attrs.get('geospatial_lon_units'))
        self.assertAlmostEqual(-62.293519314442854, ds2.attrs.get('geospatial_lat_min'))
        self.assertAlmostEqual(-62.299510171081884, ds2.attrs.get('geospatial_lat_max'))
        self.assertAlmostEqual(0.0011981728729608676, ds2.attrs.get('geospatial_lat_resolution'))
        self.assertEqual('degrees_north', ds2.attrs.get('geospatial_lat_units'))
        self.assertEqual('2018-06-01T00:00:00.000000000',
                         ds2.attrs.get('time_coverage_start'))
        self.assertEqual('2018-06-06T00:00:00.000000000',
                         ds2.attrs.get('time_coverage_end'))
        self.assertIn('date_modified', ds2.attrs)

        ds1 = xr.Dataset(coords=coords_with_bnds)
        ds1.rio.write_crs("epsg:3031",
                          inplace=True,
                          grid_mapping_name="crs").reset_coords()
        ds2 = update_dataset_attrs(ds1, global_attrs=output_metadata)

        self.assertIsNot(ds2, ds1)
        self.assertEqual('CF-1.7', ds2.attrs.get('Conventions'))
        self.assertEqual('MIT', ds2.attrs.get('license'))
        self.assertEqual('pipo', ds2.attrs.get('history'))
        self.assertAlmostEqual(-121.99380296455976, ds2.attrs.get('geospatial_lon_min'))
        self.assertAlmostEqual(-121.99083040333379, ds2.attrs.get('geospatial_lon_max'))
        self.assertAlmostEqual(0.0005945389425789926, ds2.attrs.get('geospatial_lon_resolution'))
        self.assertEqual('degrees_east', ds2.attrs.get('geospatial_lon_units'))
        self.assertAlmostEqual(-62.293519314442854, ds2.attrs.get('geospatial_lat_min'))
        self.assertAlmostEqual(-62.299510171081884, ds2.attrs.get('geospatial_lat_max'))
        self.assertAlmostEqual(0.0011981728729608676, ds2.attrs.get('geospatial_lat_resolution'))
        self.assertEqual('degrees_north', ds2.attrs.get('geospatial_lat_units'))
        self.assertEqual('2018-06-01T00:00:00.000000000',
                         ds2.attrs.get('time_coverage_start'))
        self.assertEqual('2018-06-05T23:00:59.000000000',
                         ds2.attrs.get('time_coverage_end'))
        self.assertIn('date_modified', ds2.attrs)

    def test_update_spatial_attrs(self):
        old_attrs = dict(
            geospatial_lon_min=1,
            geospatial_lon_max=2,
            geospatial_lat_min=3,
            geospatial_lat_max=4,
        )
        new_attrs = dict(
            geospatial_lon_min=-20,
            geospatial_lon_max=-18,
            geospatial_lat_min=12,
            geospatial_lat_max=13.5,
        )

        def issubdict(dict1: dict, dict2: dict) -> bool:
            return set(dict1.items()).issubset(set(dict2.items()))

        def assert_has_attrs(ds: xr.Dataset, attrs: dict) -> bool:
            self.assertTrue(issubdict(attrs, ds.attrs))

        ds1 = xr.Dataset(coords=self._create_coords()[0], attrs={})
        assert_has_attrs(update_dataset_spatial_attrs(ds1), new_attrs)

        ds2 = xr.Dataset(coords=self._create_coords()[0], attrs=old_attrs)
        assert_has_attrs(update_dataset_spatial_attrs(ds2), old_attrs)

        ds3 = xr.Dataset(coords=self._create_coords()[0], attrs=old_attrs)
        assert_has_attrs(
            update_dataset_spatial_attrs(ds3, update_existing=True),
            new_attrs
        )

        incomplete_attrs = {
            k: v for k, v in old_attrs.items() if k != 'geospatial_lon_min'
        }
        ds4 = xr.Dataset(
            coords=self._create_coords()[0], attrs=incomplete_attrs
        )
        assert_has_attrs(update_dataset_spatial_attrs(ds4), new_attrs)
