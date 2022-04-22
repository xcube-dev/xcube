import unittest

import numpy as np
import pandas as pd
import xarray as xr
from pandas import DatetimeTZDtype

from xcube.util.labels import ensure_time_compatible


class EnsureTimeCompatibleTest(unittest.TestCase):
    # A DataArray with a datetime64 time dimension -- actually implicitly
    # timezone-aware per the numpy docs, but treated as timzone-naive by
    # pandas and xarray.
    da_datetime64 = xr.DataArray(
        np.arange(1, 4),
        coords=dict(time=np.arange('2000-01-01', '2000-01-04',
                                   dtype=np.datetime64)),
        dims=['time'])

    # Currently, pd.date_range seems to produce datetime64 co-ordinates, but
    # may as well test this as a distinct case in case it changes in future.
    da_tznaive = xr.DataArray(
        np.arange(1, 4),
        coords=dict(time=pd.date_range('2000-01-01', '2000-01-03', tz=None)),
        dims=['time'])

    # To get a timezone-aware array, we use a DatetimeArray with an explicit
    # DatetimeTZDtype dtype
    da_tzaware = xr.DataArray(
        np.arange(1, 4),
        coords=dict(time=pd.arrays.DatetimeArray(
            pd.date_range('2000-01-01T00:00:00', '2000-01-03T00:00:00',
                          tz='CET'),
            dtype=DatetimeTZDtype(tz='CET'))),
        dims=['time'])
    labels_tznaive = dict(time='2000-01-02')
    labels_tzaware = dict(time='2000-01-02T00:00:00Z')

    def test_dt64_array_tznaive_indexer(self):
        self.assertEqual(self.labels_tznaive,
                         ensure_time_compatible(self.da_datetime64,
                                                self.labels_tznaive))

    def test_tzaware_array_tzaware_indexer(self):
        self.assertEqual(self.labels_tzaware,
                         ensure_time_compatible(self.da_tzaware,
                                                self.labels_tzaware))

    def test_dt64_array_tzaware_indexer(self):
        self.assertTrue(
            _are_times_equal(
                self.labels_tznaive,
                ensure_time_compatible(self.da_datetime64,
                                       self.labels_tzaware)))

    def test_tznaive_array_tzaware_indexer(self):
        self.assertTrue(
            _are_times_equal(
                self.labels_tznaive,
                ensure_time_compatible(self.da_tznaive,
                                       self.labels_tzaware)))


def _are_times_equal(labels1, labels2):
    return pd.Timestamp(labels1['time']) == pd.Timestamp(labels2['time'])