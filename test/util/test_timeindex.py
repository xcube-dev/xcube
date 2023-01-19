from unittest import TestCase

import pytest
import xarray as xr
import pandas as pd
from pytz import UTC
import numpy as np
from pandas import DatetimeTZDtype

from xcube.util.timeindex import ensure_time_label_compatible


class TimeIndexTest(TestCase):
    nonstandard_time_dimension_name = 'a_nonstandard_time_dimension_name'

    # A DataArray with a datetime64 time dimension -- actually implicitly
    # timezone-aware per the numpy docs, but treated as timezone-naive by
    # pandas and xarray.
    da_datetime64 = xr.DataArray(
        np.arange(1, 4),
        coords=dict(time=np.arange('2000-01-01', '2000-01-04',
                                   dtype=np.datetime64)),
        dims=['time'])

    da_datetime64_nonstandard_name = xr.DataArray(
        np.arange(1, 4),
        coords={nonstandard_time_dimension_name:
                    np.arange('2000-01-01', '2000-01-04', dtype=np.datetime64)},
        dims=[nonstandard_time_dimension_name])

    # As of pandas 1.4.3, pd.date_range seems to produce datetime64
    # co-ordinates, but may as well test this as a distinct case in case it
    # changes in future.
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
                         ensure_time_label_compatible(self.da_datetime64,
                                                      self.labels_tznaive))

    def test_dt64_array_tznaive_indexer_nonstandard_name(self):
        self.assertEqual(
            self.labels_tznaive,
            ensure_time_label_compatible(
                self.da_datetime64_nonstandard_name,
                self.labels_tznaive,
                self.nonstandard_time_dimension_name
            ))

    def test_dt64_array_tzaware_indexer(self):
        self.assertTrue(
            _are_times_equal(
                self.labels_tznaive,
                ensure_time_label_compatible(self.da_datetime64,
                                             self.labels_tzaware)))

    def test_tznaive_array_tzaware_indexer(self):
        self.assertTrue(
            _are_times_equal(
                self.labels_tznaive,
                ensure_time_label_compatible(self.da_tznaive,
                                             self.labels_tzaware)))

    def test_ensure_time_label_compatible_no_time(self):
        old_labels = dict(x=1)
        new_labels = ensure_time_label_compatible(
            xr.DataArray([[1, 2], [3, 4]], dims=('x', 'time')),
            old_labels
        )
        self.assertEqual(old_labels, new_labels)

    def test_ensure_time_label_compatible_no_timezone_info(self):
        old_labels = dict(time='foo')
        with pytest.warns(UserWarning):
            new_labels = ensure_time_label_compatible(
                xr.DataArray([[1, 2], [3, 4]], dims=('x', 'time'),
                             coords=dict(time=['foo', 'bar'])),
                old_labels
            )
        self.assertEqual(old_labels, new_labels)

    def test_ensure_time_label_compatible_no_tz_convert(self):
        class AwkwardTime:
            tzinfo = UTC

        old_labels = dict(time=AwkwardTime())
        time_coords = [
            pd.Timestamp('2020-01-01T12:00:00'),
            pd.Timestamp('2020-01-02T12:00:00')
        ]
        with pytest.warns(UserWarning):
            new_labels = ensure_time_label_compatible(
                xr.DataArray([[1, 2], [3, 4]], dims=('x', 'time'),
                             coords=dict(time=time_coords)),
                old_labels
            )
        self.assertEqual(old_labels, new_labels)

    def test_ensure_time_label_compatible_no_tz_localize(self):
        class AwkwardTime:
            tzinfo = None

        old_labels = dict(time=AwkwardTime())
        time_coords = [
            pd.Timestamp('2020-01-01T12:00:00+00:00'),
            pd.Timestamp('2020-01-02T12:00:00+00:00')
        ]
        with pytest.warns(UserWarning):
            new_labels = ensure_time_label_compatible(
                xr.DataArray([[1, 2], [3, 4]], dims=('x', 'time'),
                             coords=dict(time=time_coords)),
                old_labels
            )
        self.assertEqual(old_labels, new_labels)

    def test_ensure_time_label_compatible_tz_localize(self):
        old_labels = dict(time=pd.Timestamp('2020-01-01T12:00:00'))
        time_coords = [
            pd.Timestamp('2020-01-01T12:00:00+00:00'),
            pd.Timestamp('2020-01-02T12:00:00+00:00')
        ]
        new_labels = ensure_time_label_compatible(
            xr.DataArray([[1, 2], [3, 4]], dims=('x', 'time'),
                         coords=dict(time=time_coords)),
            old_labels
        )
        self.assertEqual(dict(time=pd.Timestamp('2020-01-01T12:00:00+00:00')),
                         new_labels)

    def test_with_ndarray_time_label(self):
        old_labels = dict(time=np.array(pd.Timestamp('2020-01-01T12:00:00')))
        time_coords = [
            pd.Timestamp('2020-01-01T12:00:00+00:00'),
            pd.Timestamp('2020-01-02T12:00:00+00:00')
        ]
        new_labels = ensure_time_label_compatible(
            xr.DataArray([[1, 2], [3, 4]], dims=('x', 'time'),
                         coords=dict(time=time_coords)),
            old_labels
        )
        self.assertEqual(
            dict(time=np.array(pd.Timestamp('2020-01-01T12:00:00+00:00'))),
            new_labels
        )


def _are_times_equal(labels1, labels2):
    return pd.Timestamp(labels1['time']) == pd.Timestamp(labels2['time'])
