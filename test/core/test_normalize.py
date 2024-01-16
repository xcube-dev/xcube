"""
Test for the normalization operation
"""

from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from jdcal import gcal2jd
from numpy.testing import assert_array_almost_equal

from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.core.normalize import DatasetIsNotACubeError
from xcube.core.normalize import adjust_spatial_attrs
from xcube.core.normalize import decode_cube
from xcube.core.normalize import encode_cube
from xcube.core.normalize import normalize_coord_vars
from xcube.core.normalize import normalize_dataset
from xcube.core.normalize import normalize_missing_time
from xcube.constants import CRS_CRS84


# noinspection PyPep8Naming
def assertDatasetEqual(expected, actual):
    # this method is functionally equivalent to
    # `assert expected == actual`, but it
    # checks each aspect of equality separately for easier debugging
    assert expected.equals(actual), (expected, actual)


class DecodeCubeTest(TestCase):
    def test_cube_stays_cube(self):
        dataset = new_cube(variables=dict(a=1, b=2, c=3))
        cube, grid_mapping, rest = decode_cube(dataset)
        self.assertIs(dataset, cube)
        self.assertIsInstance(grid_mapping, GridMapping)
        self.assertTrue(grid_mapping.crs.is_geographic)
        self.assertIsInstance(rest, xr.Dataset)
        self.assertEqual(set(), set(rest.data_vars))

    def test_no_cube_vars_are_dropped(self):
        dataset = new_cube(variables=dict(a=1, b=2, c=3))
        dataset = dataset.assign(
            d=xr.DataArray([8, 9, 10], dims='level'),
            crs=xr.DataArray(0, attrs=CRS_CRS84.to_cf()),
        )
        self.assertEqual({'a', 'b', 'c', 'd', 'crs'}, set(dataset.data_vars))
        cube, grid_mapping, rest = decode_cube(dataset)
        self.assertIsInstance(cube, xr.Dataset)
        self.assertIsInstance(grid_mapping, GridMapping)
        self.assertEqual({'a', 'b', 'c'}, set(cube.data_vars))
        self.assertEqual(CRS_CRS84, grid_mapping.crs)
        self.assertIsInstance(rest, xr.Dataset)
        self.assertEqual({'d', 'crs'}, set(rest.data_vars))

    def test_encode_is_inverse(self):
        dataset = new_cube(variables=dict(a=1, b=2, c=3),
                           x_name='x', y_name='y')
        dataset = dataset.assign(
            d=xr.DataArray([8, 9, 10], dims='level'),
            crs=xr.DataArray(0, attrs=CRS_CRS84.to_cf()),
        )
        cube, grid_mapping, rest = decode_cube(dataset)
        dataset2 = encode_cube(cube, grid_mapping, rest)
        self.assertEqual(set(dataset.data_vars), set(dataset2.data_vars))
        self.assertIn('crs', dataset2.data_vars)

    def test_no_cube_vars_found(self):
        dataset = new_cube()
        self.assertEqual(set(), set(dataset.data_vars))
        with self.assertRaises(DatasetIsNotACubeError) as cm:
            decode_cube(dataset, force_non_empty=True)
        self.assertEqual("No variables found with dimensions"
                         " ('time', [...] 'lat', 'lon')"
                         " or dimension sizes too small",
                         f'{cm.exception}')

    def test_no_grid_mapping(self):
        dataset = xr.Dataset(dict(a=[1, 2, 3], b=0.5))
        with self.assertRaises(DatasetIsNotACubeError) as cm:
            decode_cube(dataset)
        self.assertEqual("Failed to detect grid mapping:"
                         " cannot find any grid mapping in dataset",
                         f'{cm.exception}')

    def test_grid_mapping_not_geographic(self):
        dataset = new_cube(x_name='x', y_name='y',
                           variables=dict(a=0.5), crs='epsg:25832')
        with self.assertRaises(DatasetIsNotACubeError) as cm:
            decode_cube(dataset, force_geographic=True)
        self.assertEqual("Grid mapping must use geographic CRS,"
                         " but was 'ETRS89 / UTM zone 32N'",
                         f'{cm.exception}')


class EncodeCubeTest(TestCase):
    def test_geographical_crs(self):
        cube = new_cube(variables=dict(a=1, b=2, c=3))
        gm = GridMapping.from_dataset(cube)

        dataset = encode_cube(cube, gm)
        self.assertIs(cube, dataset)

        dataset = encode_cube(cube, gm,
                              xr.Dataset(dict(d=True)))
        self.assertIsInstance(dataset, xr.Dataset)
        self.assertEqual({'a', 'b', 'c', 'd'}, set(dataset.data_vars))

    def test_non_geographical_crs(self):
        cube = new_cube(x_name='x',
                        y_name='y',
                        crs='epsg:25832',
                        variables=dict(a=1, b=2, c=3))
        gm = GridMapping.from_dataset(cube)
        dataset = encode_cube(cube,
                              gm,
                              xr.Dataset(dict(d=True)))
        self.assertIsInstance(dataset, xr.Dataset)
        self.assertEqual({'a', 'b', 'c', 'd', 'crs'}, set(dataset.data_vars))


class TestNormalize(TestCase):

    def test_normalize_zonal_lat_lon(self):
        resolution = 10
        lat_size = 3
        lat_coords = np.arange(0, 30, resolution)
        lon_coords = [i + 5. for i in np.arange(-180.0, 180.0, resolution)]
        lon_size = len(lon_coords)
        one_more_dim_size = 2
        one_more_dim_coords = np.random.random(2)

        var_values_1_1d = xr.DataArray(np.random.random(lat_size),
                                       coords=[('latitude_centers', lat_coords)],
                                       dims=['latitude_centers'],
                                       attrs=dict(chunk_sizes=[lat_size],
                                                  dimensions=['latitude_centers']))
        var_values_1_1d.encoding = {'chunks': (lat_size,)}
        var_values_1_2d = xr.DataArray(np.array([var_values_1_1d.values for _ in lon_coords]).T,
                                       coords={'lat': lat_coords, 'lon': lon_coords},
                                       dims=['lat', 'lon'],
                                       attrs=dict(chunk_sizes=[lat_size, lon_size],
                                                  dimensions=['lat', 'lon']))
        var_values_1_2d.encoding = {'chunks': (lat_size, lon_size)}
        var_values_2_2d = xr.DataArray(np.random.random(lat_size * one_more_dim_size).
                                       reshape(lat_size, one_more_dim_size),
                                       coords={'latitude_centers': lat_coords,
                                               'one_more_dim': one_more_dim_coords},
                                       dims=['latitude_centers', 'one_more_dim'],
                                       attrs=dict(chunk_sizes=[lat_size, one_more_dim_size],
                                                  dimensions=['latitude_centers', 'one_more_dim']))
        var_values_2_2d.encoding = {'chunks': (lat_size, one_more_dim_size)}
        var_values_2_3d = xr.DataArray(np.array([var_values_2_2d.values for _ in lon_coords]).T,
                                       coords={'one_more_dim': one_more_dim_coords,
                                               'lat': lat_coords,
                                               'lon': lon_coords, },
                                       dims=['one_more_dim', 'lat', 'lon', ],
                                       attrs=dict(chunk_sizes=[one_more_dim_size,
                                                               lat_size,
                                                               lon_size],
                                                  dimensions=['one_more_dim', 'lat', 'lon']))
        var_values_2_3d.encoding = {'chunks': (one_more_dim_size, lat_size, lon_size)}

        dataset = xr.Dataset({'first': var_values_1_1d, 'second': var_values_2_2d})
        expected = xr.Dataset({'first': var_values_1_2d, 'second': var_values_2_3d})
        expected = expected.assign_coords(
            lon_bnds=xr.DataArray([[i - (resolution / 2), i + (resolution / 2)] for i in expected.lon.values],
                                  dims=['lon', 'bnds']))
        expected = expected.assign_coords(
            lat_bnds=xr.DataArray([[i - (resolution / 2), i + (resolution / 2)] for i in expected.lat.values],
                                  dims=['lat', 'bnds']))
        actual = normalize_dataset(dataset)

        xr.testing.assert_equal(actual, expected)
        self.assertEqual(actual.first.chunk_sizes, expected.first.chunk_sizes)
        self.assertEqual(actual.second.chunk_sizes, expected.second.chunk_sizes)

    def test_normalize_lon_lat_2d(self):
        """
        Test nominal execution
        """
        dims = ('time', 'y', 'x')
        attrs = {'valid_min': 0., 'valid_max': 1.}

        t_size = 2
        y_size = 3
        x_size = 4

        a_data = np.random.random_sample((t_size, y_size, x_size))
        b_data = np.random.random_sample((t_size, y_size, x_size))
        time_data = [1, 2]
        lat_data = [[10., 10., 10., 10.],
                    [20., 20., 20., 20.],
                    [30., 30., 30., 30.]]
        lon_data = [[-10., 0., 10., 20.],
                    [-10., 0., 10., 20.],
                    [-10., 0., 10., 20.]]
        dataset = xr.Dataset({'a': (dims, a_data, attrs),
                              'b': (dims, b_data, attrs)
                              },
                             {'time': (('time',), time_data),
                              'lat': (('y', 'x'), lat_data),
                              'lon': (('y', 'x'), lon_data)
                              },
                             {'geospatial_lon_min': -15.,
                              'geospatial_lon_max': 25.,
                              'geospatial_lat_min': 5.,
                              'geospatial_lat_max': 35.
                              }
                             )

        new_dims = ('time', 'lat', 'lon')
        expected = xr.Dataset({'a': (new_dims, a_data, attrs),
                               'b': (new_dims, b_data, attrs)},
                              {'time': (('time',), time_data),
                               'lat': (('lat',), [10., 20., 30.]),
                               'lon': (('lon',), [-10., 0., 10., 20.]),
                               },
                              {'geospatial_lon_min': -15.,
                               'geospatial_lon_max': 25.,
                               'geospatial_lat_min': 5.,
                               'geospatial_lat_max': 35.})

        actual = normalize_dataset(dataset)
        xr.testing.assert_equal(actual, expected)

    def test_normalize_lon_lat(self):
        """
        Test nominal execution
        """
        dataset = xr.Dataset({'first': (['latitude',
                                         'longitude'], [[1, 2, 3],
                                                        [2, 3, 4]])})
        expected = xr.Dataset({'first': (['lat', 'lon'], [[1, 2, 3],
                                                          [2, 3, 4]])})
        actual = normalize_dataset(dataset)
        assertDatasetEqual(actual, expected)

        dataset = xr.Dataset({'first': (['lat', 'long'], [[1, 2, 3],
                                                          [2, 3, 4]])})
        expected = xr.Dataset({'first': (['lat', 'lon'], [[1, 2, 3],
                                                          [2, 3, 4]])})
        actual = normalize_dataset(dataset)
        assertDatasetEqual(actual, expected)

        dataset = xr.Dataset({'first': (['latitude',
                                         'spacetime'], [[1, 2, 3],
                                                        [2, 3, 4]])})
        expected = xr.Dataset({'first': (['lat', 'spacetime'], [[1, 2, 3],
                                                                [2, 3, 4]])})
        actual = normalize_dataset(dataset)
        assertDatasetEqual(actual, expected)

        dataset = xr.Dataset({'first': (['zef', 'spacetime'], [[1, 2, 3],
                                                               [2, 3, 4]])})
        expected = xr.Dataset({'first': (['zef', 'spacetime'], [[1, 2, 3],
                                                                [2, 3, 4]])})
        actual = normalize_dataset(dataset)
        assertDatasetEqual(actual, expected)

    def test_normalize_does_not_reorder_increasing_lat(self):
        first = np.zeros([3, 45, 90])
        first[0, :, :] = np.eye(45, 90)
        ds = xr.Dataset({
            'first': (['time', 'lat', 'lon'], first),
            'second': (['time', 'lat', 'lon'], np.zeros([3, 45, 90])),
            'lat': np.linspace(-88, 88, 45),
            'lon': np.linspace(-178, 178, 90),
            'time': [datetime(2000, x, 1) for x in range(1, 4)]}).chunk(
            chunks={'time': 1})

        actual = normalize_dataset(ds)
        xr.testing.assert_equal(actual, ds)

    def test_normalize_with_missing_time_dim(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)},
                        attrs={'time_coverage_start': '20120101',
                               'time_coverage_end': '20121231'})
        norm_ds = normalize_dataset(ds)
        self.assertIsNot(norm_ds, ds)
        self.assertEqual(len(norm_ds.coords), 4)
        self.assertIn('lon', norm_ds.coords)
        self.assertIn('lat', norm_ds.coords)
        self.assertIn('time', norm_ds.coords)
        self.assertIn('time_bnds', norm_ds.coords)

        self.assertEqual(norm_ds.first.shape, (1, 90, 180))
        self.assertEqual(norm_ds.second.shape, (1, 90, 180))
        self.assertEqual(norm_ds.coords['time'][0], xr.DataArray(pd.to_datetime('2012-07-01T12:00:00')))
        self.assertEqual(norm_ds.coords['time_bnds'][0][0], xr.DataArray(pd.to_datetime('2012-01-01')))
        self.assertEqual(norm_ds.coords['time_bnds'][0][1], xr.DataArray(pd.to_datetime('2012-12-31')))

    def test_normalize_with_time_called_t(self):
        ds = xr.Dataset({'first': (['time', 'lat', 'lon'], np.zeros([4, 90, 180])),
                         'second': (['time', 'lat', 'lon'], np.zeros([4, 90, 180])),
                         't': ('time', np.array(['2005-07-02T00:00:00.000000000',
                                                 '2006-07-02T12:00:00.000000000',
                                                 '2007-07-03T00:00:00.000000000',
                                                 '2008-07-02T00:00:00.000000000'], dtype='datetime64[ns]'))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)},
                        attrs={'time_coverage_start': '2005-01-17',
                               'time_coverage_end': '2008-08-17'})
        norm_ds = normalize_dataset(ds)
        self.assertIsNot(norm_ds, ds)
        self.assertEqual(len(norm_ds.coords), 3)
        self.assertIn('lon', norm_ds.coords)
        self.assertIn('lat', norm_ds.coords)
        self.assertIn('time', norm_ds.coords)

        self.assertEqual(norm_ds.first.shape, (4, 90, 180))
        self.assertEqual(norm_ds.second.shape, (4, 90, 180))
        self.assertEqual(norm_ds.coords['time'][0], xr.DataArray(pd.to_datetime('2005-07-02T00:00')))

    def test_normalize_julian_day(self):
        """
        Test Julian Day -> Datetime conversion
        """
        tuples = [gcal2jd(2000, x, 1) for x in range(1, 13)]

        ds = xr.Dataset({
            'first': (['lat', 'lon', 'time'], np.zeros([88, 90, 12])),
            'second': (['lat', 'lon', 'time'], np.zeros([88, 90, 12])),
            'lat': np.linspace(-88, 45, 88),
            'lon': np.linspace(-178, 178, 90),
            'time': [x[0] + x[1] for x in tuples]})
        ds.time.attrs['long_name'] = 'time in julian days'

        expected = xr.Dataset({
            'first': (['time', 'lat', 'lon'], np.zeros([12, 88, 90])),
            'second': (['time', 'lat', 'lon'], np.zeros([12, 88, 90])),
            'lat': np.linspace(-88, 45, 88),
            'lon': np.linspace(-178, 178, 90),
            'time': [datetime(2000, x, 1) for x in range(1, 13)]})
        expected.time.attrs['long_name'] = 'time'

        actual = normalize_dataset(ds)

        assertDatasetEqual(actual, expected)


class AdjustSpatialTest(TestCase):
    def test_nominal(self):
        ds = xr.Dataset({
            'first': (['lat', 'lon', 'time'], np.zeros([45, 90, 12])),
            'second': (['lat', 'lon', 'time'], np.zeros([45, 90, 12])),
            'lat': np.linspace(-88, 88, 45),
            'lon': np.linspace(-178, 178, 90),
            'time': [datetime(2000, x, 1) for x in range(1, 13)]})

        ds.lon.attrs['units'] = 'degrees_east'
        ds.lat.attrs['units'] = 'degrees_north'

        ds1 = adjust_spatial_attrs(ds)

        # Make sure original dataset is not altered
        with self.assertRaises(KeyError):
            # noinspection PyStatementEffect
            ds.attrs['geospatial_lat_min']

        # Make sure expected values are in the new dataset
        self.assertEqual(ds1.attrs['geospatial_lat_min'], -90)
        self.assertEqual(ds1.attrs['geospatial_lat_max'], 90)
        self.assertEqual(ds1.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertEqual(ds1.attrs['geospatial_lat_resolution'], 4)
        self.assertEqual(ds1.attrs['geospatial_lon_min'], -180)
        self.assertEqual(ds1.attrs['geospatial_lon_max'], 180)
        self.assertEqual(ds1.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds1.attrs['geospatial_lon_resolution'], 4)
        self.assertEqual(ds1.attrs['geospatial_bounds'],
                         'POLYGON((-180.0 -90.0, -180.0 90.0, 180.0 90.0,'
                         ' 180.0 -90.0, -180.0 -90.0))')

        # Test existing attributes update
        lon_min, lat_min, lon_max, lat_max = -20, -40, 60, 40
        indexers = {'lon': slice(lon_min, lon_max),
                    'lat': slice(lat_min, lat_max)}
        ds2 = ds1.sel(**indexers)
        ds2 = adjust_spatial_attrs(ds2)

        self.assertEqual(ds2.attrs['geospatial_lat_min'], -42)
        self.assertEqual(ds2.attrs['geospatial_lat_max'], 42)
        self.assertEqual(ds2.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertEqual(ds2.attrs['geospatial_lat_resolution'], 4)
        self.assertEqual(ds2.attrs['geospatial_lon_min'], -20)
        self.assertEqual(ds2.attrs['geospatial_lon_max'], 60)
        self.assertEqual(ds2.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds2.attrs['geospatial_lon_resolution'], 4)
        self.assertEqual(ds2.attrs['geospatial_bounds'],
                         'POLYGON((-20.0 -42.0, -20.0 42.0, 60.0 42.0, 60.0'
                         ' -42.0, -20.0 -42.0))')

    def test_nominal_inverted(self):
        # Inverted lat
        ds = xr.Dataset({
            'first': (['lat', 'lon', 'time'], np.zeros([45, 90, 12])),
            'second': (['lat', 'lon', 'time'], np.zeros([45, 90, 12])),
            'lat': np.linspace(88, -88, 45),
            'lon': np.linspace(-178, 178, 90),
            'time': [datetime(2000, x, 1) for x in range(1, 13)]})

        ds.lon.attrs['units'] = 'degrees_east'
        ds.lat.attrs['units'] = 'degrees_north'

        ds1 = adjust_spatial_attrs(ds)

        # Make sure original dataset is not altered
        with self.assertRaises(KeyError):
            # noinspection PyStatementEffect
            ds.attrs['geospatial_lat_min']

        # Make sure expected values are in the new dataset
        self.assertEqual(ds1.attrs['geospatial_lat_min'], -90)
        self.assertEqual(ds1.attrs['geospatial_lat_max'], 90)
        self.assertEqual(ds1.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertEqual(ds1.attrs['geospatial_lat_resolution'], 4)
        self.assertEqual(ds1.attrs['geospatial_lon_min'], -180)
        self.assertEqual(ds1.attrs['geospatial_lon_max'], 180)
        self.assertEqual(ds1.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds1.attrs['geospatial_lon_resolution'], 4)
        self.assertEqual(ds1.attrs['geospatial_bounds'],
                         'POLYGON((-180.0 -90.0, -180.0 90.0, 180.0 90.0,'
                         ' 180.0 -90.0, -180.0 -90.0))')

        # Test existing attributes update
        lon_min, lat_min, lon_max, lat_max = -20, -40, 60, 40
        indexers = {'lon': slice(lon_min, lon_max),
                    'lat': slice(lat_max, lat_min)}
        ds2 = ds1.sel(**indexers)
        ds2 = adjust_spatial_attrs(ds2)

        self.assertEqual(ds2.attrs['geospatial_lat_min'], -42)
        self.assertEqual(ds2.attrs['geospatial_lat_max'], 42)
        self.assertEqual(ds2.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertEqual(ds2.attrs['geospatial_lat_resolution'], 4)
        self.assertEqual(ds2.attrs['geospatial_lon_min'], -20)
        self.assertEqual(ds2.attrs['geospatial_lon_max'], 60)
        self.assertEqual(ds2.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds2.attrs['geospatial_lon_resolution'], 4)
        self.assertEqual(ds2.attrs['geospatial_bounds'],
                         'POLYGON((-20.0 -42.0, -20.0 42.0, 60.0 42.0, 60.0'
                         ' -42.0, -20.0 -42.0))')

    def test_bnds(self):
        ds = xr.Dataset({
            'first': (['lat', 'lon', 'time'], np.zeros([45, 90, 12])),
            'second': (['lat', 'lon', 'time'], np.zeros([45, 90, 12])),
            'lat': np.linspace(-88, 88, 45),
            'lon': np.linspace(-178, 178, 90),
            'time': [datetime(2000, x, 1) for x in range(1, 13)]})

        ds.lon.attrs['units'] = 'degrees_east'
        ds.lat.attrs['units'] = 'degrees_north'

        lat_bnds = np.empty([len(ds.lat), 2])
        lon_bnds = np.empty([len(ds.lon), 2])
        ds['nv'] = [0, 1]

        lat_bnds[:, 0] = ds.lat.values - 2
        lat_bnds[:, 1] = ds.lat.values + 2
        lon_bnds[:, 0] = ds.lon.values - 2
        lon_bnds[:, 1] = ds.lon.values + 2

        ds['lat_bnds'] = (['lat', 'nv'], lat_bnds)
        ds['lon_bnds'] = (['lon', 'nv'], lon_bnds)

        ds.lat.attrs['bounds'] = 'lat_bnds'
        ds.lon.attrs['bounds'] = 'lon_bnds'

        ds1 = adjust_spatial_attrs(ds)

        # Make sure original dataset is not altered
        with self.assertRaises(KeyError):
            # noinspection PyStatementEffect
            ds.attrs['geospatial_lat_min']

        # Make sure expected values are in the new dataset
        self.assertEqual(ds1.attrs['geospatial_lat_min'], -90)
        self.assertEqual(ds1.attrs['geospatial_lat_max'], 90)
        self.assertEqual(ds1.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertEqual(ds1.attrs['geospatial_lat_resolution'], 4)
        self.assertEqual(ds1.attrs['geospatial_lon_min'], -180)
        self.assertEqual(ds1.attrs['geospatial_lon_max'], 180)
        self.assertEqual(ds1.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds1.attrs['geospatial_lon_resolution'], 4)
        self.assertEqual(ds1.attrs['geospatial_bounds'],
                         'POLYGON((-180.0 -90.0, -180.0 90.0, 180.0 90.0,'
                         ' 180.0 -90.0, -180.0 -90.0))')

        # Test existing attributes update
        lon_min, lat_min, lon_max, lat_max = -20, -40, 60, 40
        indexers = {'lon': slice(lon_min, lon_max),
                    'lat': slice(lat_min, lat_max)}
        ds2 = ds1.sel(**indexers)
        ds2 = adjust_spatial_attrs(ds2)

        self.assertEqual(ds2.attrs['geospatial_lat_min'], -42)
        self.assertEqual(ds2.attrs['geospatial_lat_max'], 42)
        self.assertEqual(ds2.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertEqual(ds2.attrs['geospatial_lat_resolution'], 4)
        self.assertEqual(ds2.attrs['geospatial_lon_min'], -20)
        self.assertEqual(ds2.attrs['geospatial_lon_max'], 60)
        self.assertEqual(ds2.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds2.attrs['geospatial_lon_resolution'], 4)
        self.assertEqual(ds2.attrs['geospatial_bounds'],
                         'POLYGON((-20.0 -42.0, -20.0 42.0, 60.0 42.0, 60.0'
                         ' -42.0, -20.0 -42.0))')

    def test_bnds_inverted(self):
        # Inverted lat
        ds = xr.Dataset({
            'first': (['lat', 'lon', 'time'], np.zeros([45, 90, 12])),
            'second': (['lat', 'lon', 'time'], np.zeros([45, 90, 12])),
            'lat': np.linspace(88, -88, 45),
            'lon': np.linspace(-178, 178, 90),
            'time': [datetime(2000, x, 1) for x in range(1, 13)]})

        ds.lon.attrs['units'] = 'degrees_east'
        ds.lat.attrs['units'] = 'degrees_north'

        lat_bnds = np.empty([len(ds.lat), 2])
        lon_bnds = np.empty([len(ds.lon), 2])
        ds['nv'] = [0, 1]

        lat_bnds[:, 0] = ds.lat.values + 2
        lat_bnds[:, 1] = ds.lat.values - 2
        lon_bnds[:, 0] = ds.lon.values - 2
        lon_bnds[:, 1] = ds.lon.values + 2

        ds['lat_bnds'] = (['lat', 'nv'], lat_bnds)
        ds['lon_bnds'] = (['lon', 'nv'], lon_bnds)

        ds.lat.attrs['bounds'] = 'lat_bnds'
        ds.lon.attrs['bounds'] = 'lon_bnds'

        ds1 = adjust_spatial_attrs(ds)

        # Make sure original dataset is not altered
        with self.assertRaises(KeyError):
            # noinspection PyStatementEffect
            ds.attrs['geospatial_lat_min']

        # Make sure expected values are in the new dataset
        self.assertEqual(ds1.attrs['geospatial_lat_min'], -90)
        self.assertEqual(ds1.attrs['geospatial_lat_max'], 90)
        self.assertEqual(ds1.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertEqual(ds1.attrs['geospatial_lat_resolution'], 4)
        self.assertEqual(ds1.attrs['geospatial_lon_min'], -180)
        self.assertEqual(ds1.attrs['geospatial_lon_max'], 180)
        self.assertEqual(ds1.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds1.attrs['geospatial_lon_resolution'], 4)
        self.assertEqual(ds1.attrs['geospatial_bounds'],
                         'POLYGON((-180.0 -90.0, -180.0 90.0, 180.0 90.0,'
                         ' 180.0 -90.0, -180.0 -90.0))')

        # Test existing attributes update
        lon_min, lat_min, lon_max, lat_max = -20, -40, 60, 40
        indexers = {'lon': slice(lon_min, lon_max),
                    'lat': slice(lat_max, lat_min)}
        ds2 = ds1.sel(**indexers)
        ds2 = adjust_spatial_attrs(ds2)

        self.assertEqual(ds2.attrs['geospatial_lat_min'], -42)
        self.assertEqual(ds2.attrs['geospatial_lat_max'], 42)
        self.assertEqual(ds2.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertEqual(ds2.attrs['geospatial_lat_resolution'], 4)
        self.assertEqual(ds2.attrs['geospatial_lon_min'], -20)
        self.assertEqual(ds2.attrs['geospatial_lon_max'], 60)
        self.assertEqual(ds2.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds2.attrs['geospatial_lon_resolution'], 4)
        self.assertEqual(ds2.attrs['geospatial_bounds'],
                         'POLYGON((-20.0 -42.0, -20.0 42.0, 60.0 42.0, 60.0 -42.0, -20.0 -42.0))')

    def test_once_cell_with_bnds(self):
        # Only one cell in lat/lon
        ds = xr.Dataset({
            'first': (['lat', 'lon', 'time'], np.zeros([1, 1, 12])),
            'second': (['lat', 'lon', 'time'], np.zeros([1, 1, 12])),
            'lat': np.array([52.5]),
            'lon': np.array([11.5]),
            'lat_bnds': (['lat', 'bnds'], np.array([[52.4, 52.6]])),
            'lon_bnds': (['lon', 'bnds'], np.array([[11.4, 11.6]])),
            'time': [datetime(2000, x, 1) for x in range(1, 13)]})
        ds.lon.attrs['units'] = 'degrees_east'
        ds.lat.attrs['units'] = 'degrees_north'

        ds1 = adjust_spatial_attrs(ds)
        self.assertAlmostEqual(ds1.attrs['geospatial_lat_resolution'], 0.2)
        self.assertAlmostEqual(ds1.attrs['geospatial_lat_min'], 52.4)
        self.assertAlmostEqual(ds1.attrs['geospatial_lat_max'], 52.6)
        self.assertEqual(ds1.attrs['geospatial_lat_units'], 'degrees_north')
        self.assertAlmostEqual(ds1.attrs['geospatial_lon_resolution'], 0.2)
        self.assertAlmostEqual(ds1.attrs['geospatial_lon_min'], 11.4)
        self.assertAlmostEqual(ds1.attrs['geospatial_lon_max'], 11.6)
        self.assertEqual(ds1.attrs['geospatial_lon_units'], 'degrees_east')
        self.assertEqual(ds1.attrs['geospatial_bounds'],
                         'POLYGON((11.4 52.4, 11.4 52.6, 11.6 52.6, 11.6 52.4, 11.4 52.4))')

    def test_once_cell_without_bnds(self):
        # Only one cell in lat/lon
        ds = xr.Dataset({
            'first': (['lat', 'lon', 'time'], np.zeros([1, 1, 12])),
            'second': (['lat', 'lon', 'time'], np.zeros([1, 1, 12])),
            'lat': np.array([52.5]),
            'lon': np.array([11.5]),
            'time': [datetime(2000, x, 1) for x in range(1, 13)]})
        ds.lon.attrs['units'] = 'degrees_east'
        ds.lat.attrs['units'] = 'degrees_north'

        ds2 = adjust_spatial_attrs(ds)
        # Datasets should be the same --> not modified
        self.assertIs(ds2, ds)


class NormalizeCoordVarsTest(TestCase):

    def test_ds_with_potential_coords(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180])),
                         'lat_bnds': (['lat', 'bnds'], np.zeros([90, 2])),
                         'lon_bnds': (['lon', 'bnds'], np.zeros([180, 2]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)})

        new_ds = normalize_coord_vars(ds)

        self.assertIsNot(ds, new_ds)
        self.assertEqual(len(new_ds.coords), 4)
        self.assertIn('lon', new_ds.coords)
        self.assertIn('lat', new_ds.coords)
        self.assertIn('lat_bnds', new_ds.coords)
        self.assertIn('lon_bnds', new_ds.coords)

        self.assertEqual(len(new_ds.data_vars), 2)
        self.assertIn('first', new_ds.data_vars)
        self.assertIn('second', new_ds.data_vars)

    def test_ds_with_potential_coords_and_bounds(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180])),
                         'lat_bnds': (['lat', 'bnds'], np.zeros([90, 2])),
                         'lon_bnds': (['lon', 'bnds'], np.zeros([180, 2])),
                         'lat': (['lat'], np.linspace(-89.5, 89.5, 90)),
                         'lon': (['lon'], np.linspace(-179.5, 179.5, 180))})

        new_ds = normalize_coord_vars(ds)

        self.assertIsNot(ds, new_ds)
        self.assertEqual(len(new_ds.coords), 4)
        self.assertIn('lon', new_ds.coords)
        self.assertIn('lat', new_ds.coords)
        self.assertIn('lat_bnds', new_ds.coords)
        self.assertIn('lon_bnds', new_ds.coords)

        self.assertEqual(len(new_ds.data_vars), 2)
        self.assertIn('first', new_ds.data_vars)
        self.assertIn('second', new_ds.data_vars)

    def test_ds_with_no_potential_coords(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)},
                        attrs={'time_coverage_start': '20120101'})
        new_ds = normalize_coord_vars(ds)
        self.assertIs(ds, new_ds)


class NormalizeMissingTimeTest(TestCase):
    def test_ds_without_time(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)},
                        attrs={'time_coverage_start': '20120101',
                               'time_coverage_end': '20121231'})

        new_ds = normalize_missing_time(ds)

        self.assertIsNot(ds, new_ds)
        self.assertEqual(len(new_ds.coords), 4)
        self.assertIn('lon', new_ds.coords)
        self.assertIn('lat', new_ds.coords)
        self.assertIn('time', new_ds.coords)
        self.assertIn('time_bnds', new_ds.coords)

        self.assertEqual(new_ds.coords['time'].attrs.get('long_name'), 'time')
        self.assertEqual(new_ds.coords['time'].attrs.get('bounds'), 'time_bnds')

        self.assertEqual(new_ds.first.shape, (1, 90, 180))
        self.assertEqual(new_ds.second.shape, (1, 90, 180))
        self.assertEqual(new_ds.coords['time'][0], xr.DataArray(pd.to_datetime('2012-07-01T12:00:00')))
        self.assertEqual(new_ds.coords['time'].attrs.get('long_name'), 'time')
        self.assertEqual(new_ds.coords['time'].attrs.get('bounds'), 'time_bnds')
        self.assertEqual(new_ds.coords['time_bnds'][0][0], xr.DataArray(pd.to_datetime('2012-01-01')))
        self.assertEqual(new_ds.coords['time_bnds'][0][1], xr.DataArray(pd.to_datetime('2012-12-31')))
        self.assertEqual(new_ds.coords['time_bnds'].attrs.get('long_name'), 'time')

    def test_ds_without_bounds(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)},
                        attrs={'time_coverage_start': '20120101'})

        new_ds = normalize_missing_time(ds)

        self.assertIsNot(ds, new_ds)
        self.assertEqual(len(new_ds.coords), 3)
        self.assertIn('lon', new_ds.coords)
        self.assertIn('lat', new_ds.coords)
        self.assertIn('time', new_ds.coords)
        self.assertNotIn('time_bnds', new_ds.coords)

        self.assertEqual(new_ds.first.shape, (1, 90, 180))
        self.assertEqual(new_ds.second.shape, (1, 90, 180))
        self.assertEqual(new_ds.coords['time'][0], xr.DataArray(pd.to_datetime('2012-01-01')))
        self.assertEqual(new_ds.coords['time'].attrs.get('long_name'), 'time')
        self.assertEqual(new_ds.coords['time'].attrs.get('bounds'), None)

    def test_ds_without_time_attrs(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)})

        new_ds = normalize_missing_time(ds)
        self.assertIs(ds, new_ds)

    def test_ds_with_cftime(self):
        time_data = xr.cftime_range(start='2010-01-01T00:00:00',
                                    periods=6,
                                    freq='D',
                                    calendar='gregorian').values
        ds = xr.Dataset({'first': (['time', 'lat', 'lon'], np.zeros([6, 90, 180])),
                         'second': (['time', 'lat', 'lon'], np.zeros([6, 90, 180]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180),
                                'time': time_data},
                        attrs={'time_coverage_start': '20120101',
                               'time_coverage_end': '20121231'})
        new_ds = normalize_missing_time(ds)
        self.assertIs(ds, new_ds)

    def test_normalize_with_missing_time_dim(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)},
                        attrs={'time_coverage_start': '20120101',
                               'time_coverage_end': '20121231'})
        norm_ds = normalize_dataset(ds)
        self.assertIsNot(norm_ds, ds)
        self.assertEqual(len(norm_ds.coords), 4)
        self.assertIn('lon', norm_ds.coords)
        self.assertIn('lat', norm_ds.coords)
        self.assertIn('time', norm_ds.coords)
        self.assertIn('time_bnds', norm_ds.coords)

        self.assertEqual(norm_ds.first.shape, (1, 90, 180))
        self.assertEqual(norm_ds.second.shape, (1, 90, 180))
        self.assertEqual(norm_ds.coords['time'][0], xr.DataArray(pd.to_datetime('2012-07-01T12:00:00')))
        self.assertEqual(norm_ds.coords['time_bnds'][0][0], xr.DataArray(pd.to_datetime('2012-01-01')))
        self.assertEqual(norm_ds.coords['time_bnds'][0][1], xr.DataArray(pd.to_datetime('2012-12-31')))

    def test_normalize_with_missing_time_dim_from_filename(self):
        ds = xr.Dataset({'first': (['lat', 'lon'], np.zeros([90, 180])),
                         'second': (['lat', 'lon'], np.zeros([90, 180]))},
                        coords={'lat': np.linspace(-89.5, 89.5, 90),
                                'lon': np.linspace(-179.5, 179.5, 180)},
                        )
        ds_encoding = dict(source='20150204_etfgz_20170309_dtsrgth')
        ds.encoding.update(ds_encoding)
        norm_ds = normalize_dataset(ds)
        self.assertIsNot(norm_ds, ds)
        self.assertEqual(len(norm_ds.coords), 4)
        self.assertIn('lon', norm_ds.coords)
        self.assertIn('lat', norm_ds.coords)
        self.assertIn('time', norm_ds.coords)
        self.assertIn('time_bnds', norm_ds.coords)

        self.assertEqual(norm_ds.first.shape, (1, 90, 180))
        self.assertEqual(norm_ds.second.shape, (1, 90, 180))
        self.assertEqual(norm_ds.coords['time'][0], xr.DataArray(pd.to_datetime('2016-02-21T00:00:00')))
        self.assertEqual(norm_ds.coords['time_bnds'][0][0], xr.DataArray(pd.to_datetime('2015-02-04')))
        self.assertEqual(norm_ds.coords['time_bnds'][0][1], xr.DataArray(pd.to_datetime('2017-03-09')))


class Fix360Test(TestCase):

    def test_fix_360_lon(self):
        # The following simulates a strangely geo-coded soil moisture dataset we found
        lon_size = 360
        lat_size = 130
        time_size = 12
        ds = xr.Dataset({
            'first': (['time', 'lat', 'lon'],
                      np.random.random_sample([time_size, lat_size, lon_size])),
            'second': (['time', 'lat', 'lon'],
                       np.random.random_sample([time_size, lat_size,
                                                lon_size]))},
            coords={'lon': np.linspace(1., 360., lon_size),
                    'lat': np.linspace(-65., 64., lat_size),
                    'time': [datetime(2000, x, 1)
                             for x in range(1, time_size + 1)]},
            attrs=dict(geospatial_lon_min=0.,
                       geospatial_lon_max=360.,
                       geospatial_lat_min=-65.5,
                       geospatial_lat_max=+64.5,
                       geospatial_lon_resolution=1.,
                       geospatial_lat_resolution=1.))

        new_ds = normalize_dataset(ds)
        self.assertIsNot(ds, new_ds)
        self.assertEqual(ds.dims, new_ds.dims)
        self.assertEqual(ds.sizes, new_ds.sizes)
        assert_array_almost_equal(new_ds.lon, np.linspace(-179.5, 179.5, 360))
        assert_array_almost_equal(new_ds.lat, np.linspace(-65., 64., 130))
        assert_array_almost_equal(new_ds.first[..., :180], ds.first[..., 180:])
        assert_array_almost_equal(new_ds.first[..., 180:], ds.first[..., :180])
        assert_array_almost_equal(new_ds.second[..., :180],
                                  ds.second[..., 180:])
        assert_array_almost_equal(new_ds.second[..., 180:],
                                  ds.second[..., :180])
        self.assertEqual(-180., new_ds.attrs['geospatial_lon_min'])
        self.assertEqual(+180., new_ds.attrs['geospatial_lon_max'])
        self.assertEqual(-65.5, new_ds.attrs['geospatial_lat_min'])
        self.assertEqual(+64.5, new_ds.attrs['geospatial_lat_max'])
        self.assertEqual(1., new_ds.attrs['geospatial_lon_resolution'])
        self.assertEqual(1., new_ds.attrs['geospatial_lat_resolution'])


class NormalizeDimOrderTest(TestCase):
    """
    Test normalize_cci_sea_level operation
    """

    def test_no_change(self):
        """
        Test nominal operation
        """
        lon_size = 360
        lat_size = 130
        time_size = 12
        ds = xr.Dataset({
            'first': (['time', 'lat', 'lon'],
                      np.random.random_sample([time_size, lat_size, lon_size])),
            'second': (['time', 'lat', 'lon'],
                       np.random.random_sample([time_size, lat_size,
                                                lon_size]))},
            coords={'lon': np.linspace(-179.5, -179.5, lon_size),
                    'lat': np.linspace(-65., 64., lat_size),
                    'time': [datetime(2000, x, 1)
                             for x in range(1, time_size + 1)]})
        ds2 = normalize_dataset(ds)
        self.assertIs(ds2, ds)

    def test_nominal(self):
        """
        Test nominal operation
        """
        ds = self.new_cci_seal_level_ds()
        ds2 = normalize_dataset(ds)
        self.assertIsNot(ds2, ds)
        self.assertIn('ampl', ds2)
        self.assertIn('phase', ds2)
        self.assertIn('time', ds2.coords)
        self.assertIn('time_bnds', ds2.coords)
        self.assertNotIn('time_step', ds2.coords)
        self.assertEqual(['time', 'period', 'lat', 'lon'], list(ds2.ampl.dims))
        self.assertEqual(['time', 'period', 'lat', 'lon'], list(ds2.phase.dims))

    @staticmethod
    def new_cci_seal_level_ds():
        period_size = 2
        lon_size = 4
        lat_size = 2

        dataset = xr.Dataset(dict(ampl=(['lat', 'lon', 'period'], np.ones(shape=(lat_size, lon_size, period_size))),
                                  phase=(['lat', 'lon', 'period'], np.zeros(shape=(lat_size, lon_size, period_size)))),
                             coords=dict(lon=np.array([-135, -45., 45., 135.]), lat=np.array([-45., 45.]),
                                         time=pd.to_datetime(
                                             ['1993-01-15T00:00:00.000000000', '1993-02-15T00:00:00.000000000',
                                              '2015-11-15T00:00:00.000000000', '2015-12-15T00:00:00.000000000'])))

        dataset.coords['time'].encoding.update(units='days since 1950-01-01', dtype=np.dtype(np.float32))
        dataset.coords['time'].attrs.update(long_name='time', standard_name='time')

        dataset.attrs['time_coverage_start'] = '1993-01-01 00:00:00'
        dataset.attrs['time_coverage_end'] = '2015-12-31 23:59:59'

        return dataset
