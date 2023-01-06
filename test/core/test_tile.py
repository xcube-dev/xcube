import unittest

import numpy as np
import pandas as pd
import xarray as xr

# noinspection PyProtectedMember
from xcube.core.tile import _ensure_time_compatible
from xcube.core.tile import get_var_valid_range
from xcube.core.tile import parse_non_spatial_labels


class GetVarValidRangeTest(unittest.TestCase):
    def test_from_valid_range(self):
        a = xr.DataArray(0, attrs=dict(valid_range=[-1, 1]))
        self.assertEqual((-1, 1),
                         get_var_valid_range(a))

    def test_from_valid_min_max(self):
        a = xr.DataArray(0, attrs=dict(valid_min=-1, valid_max=1))
        self.assertEqual((-1, 1),
                         get_var_valid_range(a))

    def test_from_valid_min(self):
        a = xr.DataArray(0, attrs=dict(valid_min=-1))
        self.assertEqual((-1, np.inf),
                         get_var_valid_range(a))

    def test_from_valid_max(self):
        a = xr.DataArray(0, attrs=dict(valid_max=1))
        self.assertEqual((-np.inf, 1),
                         get_var_valid_range(a))

    def test_from_nothing(self):
        a = xr.DataArray(0)
        self.assertEqual(None,
                         get_var_valid_range(a))


class ParseNonSpatialLabelsTest(unittest.TestCase):
    time = xr.DataArray(np.array(['2021-06-07 10:20:45',
                                  '2021-06-08 13:28:16',
                                  '2021-06-09 11:10:21'],
                                 dtype='datetime64'),
                        dims='time')
    lat = xr.DataArray(np.array([60, 61, 62], dtype=np.float64),
                       dims='lat')
    lon = xr.DataArray(np.array([4, 5, 6], dtype=np.float64),
                       dims='lon')
    coords = dict(time=time, lat=lat, lon=lon)

    dims = ('time', 'lat', 'lon')

    def test_use_first_not_given(self):
        labels = parse_non_spatial_labels(dict(),
                                          dims=self.dims,
                                          coords=self.coords)
        self.assertEqual(
            dict(time=np.array('2021-06-07 10:20:45', dtype='datetime64')),
            labels
        )

    def test_use_last_if_current(self):
        labels = parse_non_spatial_labels(dict(time='current'),
                                          dims=self.dims,
                                          coords=self.coords)
        self.assertEqual(
            dict(time=np.array('2021-06-09 11:10:21', dtype='datetime64')),
            labels
        )

    def test_use_given_if_value_given(self):
        labels = parse_non_spatial_labels(dict(time='2021-06-08'),
                                          dims=self.dims,
                                          coords=self.coords)
        self.assertEqual(
            dict(time=np.array('2021-06-08', dtype='datetime64')),
            labels
        )

    def test_use_average_if_range_given(self):
        labels = parse_non_spatial_labels(dict(time='2021-06-06/2021-06-08'),
                                          dims=self.dims,
                                          coords=self.coords)
        self.assertEqual(
            dict(time=np.array('2021-06-07', dtype='datetime64')),
            labels
        )

    def test_use_slice_if_range_given(self):
        labels = parse_non_spatial_labels(dict(time='2021-06-06/2021-06-08'),
                                          dims=self.dims,
                                          coords=self.coords,
                                          allow_slices=True)
        self.assertEqual(
            dict(time=slice(np.array('2021-06-06', dtype='datetime64'),
                            np.array('2021-06-08', dtype='datetime64'))),
            labels
        )

    def test_invalid(self):
        with self.assertRaises(ValueError) as cm:
            parse_non_spatial_labels(dict(time='jetzt'),
                                     dims=self.dims,
                                     coords=self.coords)
        self.assertEqual("'jetzt' is not a valid value for dimension 'time'",
                         f'{cm.exception}')

    def test_ensure_timezone_naive(self):
        da_tznaive = xr.DataArray(
            np.zeros((3, 3, 3)),
            coords=self.coords,
            dims=self.dims
        )
        labels = parse_non_spatial_labels(dict(time='2000-01-02T00:00:00Z'),
                                          dims=da_tznaive.dims,
                                          coords=da_tznaive.coords,
                                          var=da_tznaive)
        self.assertIsNone(pd.Timestamp(labels['time']).tzinfo)


class EnsureTimeCompatibleTest(unittest.TestCase):
    da_tznaive = xr.DataArray(
        np.arange(1, 4),
        coords=dict(time=np.arange('2000-01-01', '2000-01-04',
                                   dtype=np.datetime64)),
        dims=['time'])
    da_tzaware = xr.DataArray(
        np.arange(1, 4),
        coords=dict(time=pd.date_range("2000-01-01", "2000-01-03", tz='UTC')),
        dims=['time'])
    labels_tznaive = dict(time='2000-01-02')
    labels_tzaware = dict(time='2000-01-02T00:00:00Z')

    def test_both_tznaive(self):
        self.assertEqual(self.labels_tznaive,
                         _ensure_time_compatible(self.da_tznaive,
                                                 self.labels_tznaive))

    def test_both_tzaware(self):
        self.assertEqual(self.labels_tzaware,
                         _ensure_time_compatible(self.da_tzaware,
                                                 self.labels_tzaware))

    def test_tznaive_array_tzaware_indexer(self):
        self.assertTrue(
            _are_times_equal(
                self.labels_tznaive,
                _ensure_time_compatible(self.da_tznaive,
                                        self.labels_tzaware)))


def _are_times_equal(labels1, labels2):
    return pd.Timestamp(labels1['time']) == pd.Timestamp(labels2['time'])
