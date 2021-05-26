import unittest

import numpy as np
import pandas as pd

from test.sampledata import create_highroc_dataset
from xcube.core.new import new_cube
from xcube.core.timecoord import add_time_coords
from xcube.core.timecoord import from_time_in_days_since_1970
from xcube.core.timecoord import get_end_time_from_attrs
from xcube.core.timecoord import get_start_time_from_attrs
from xcube.core.timecoord import get_time_range_from_attrs
from xcube.core.timecoord import get_time_range_from_data
from xcube.core.timecoord import timestamp_to_iso_string
from xcube.core.timecoord import to_time_in_days_since_1970


class AddTimeCoordsTest(unittest.TestCase):

    def test_add_time_coords_point(self):
        dataset = create_highroc_dataset()
        dataset_with_time = add_time_coords(dataset, (365 * 47 + 20, 365 * 47 + 20))
        self.assertIsNot(dataset_with_time, dataset)
        self.assertIn('time', dataset_with_time)
        self.assertEqual(dataset_with_time.time.shape, (1,))
        self.assertNotIn('time_bnds', dataset_with_time)

    def test_add_time_coords_range(self):
        dataset = create_highroc_dataset()
        dataset_with_time = add_time_coords(dataset, (365 * 47 + 20, 365 * 47 + 21))
        self.assertIsNot(dataset_with_time, dataset)
        self.assertIn('time', dataset_with_time)
        self.assertEqual(dataset_with_time.time.shape, (1,))
        self.assertIn('time_bnds', dataset_with_time)
        self.assertEqual(dataset_with_time.time_bnds.shape, (1, 2))

    def test_to_time_in_days_since_1970(self):
        self.assertEqual(17324.5,
                         to_time_in_days_since_1970('201706071200'))
        self.assertEqual(17325.5,
                         to_time_in_days_since_1970('201706081200'))
        self.assertEqual(17690.5,
                         to_time_in_days_since_1970('2018-06-08 12:00'))
        self.assertEqual(17690.5,
                         to_time_in_days_since_1970('2018-06-08T12:00'))
        self.assertEqual(18173.42625622898,
                         to_time_in_days_since_1970('04-OCT-2019 10:13:48.538184'))

    def test_from_time_in_days_since_1970(self):
        self.assertEqual('2017-06-07T12:00:00.000000000',
                         str(from_time_in_days_since_1970(
                             to_time_in_days_since_1970('201706071200'))))
        self.assertEqual('2017-06-08T12:00:00.000000000',
                         str(from_time_in_days_since_1970(
                             to_time_in_days_since_1970('201706081200'))))
        self.assertEqual('2018-06-08T12:00:00.000000000',
                         str(from_time_in_days_since_1970(
                             to_time_in_days_since_1970('2018-06-08 12:00'))))
        self.assertEqual('2018-06-08T12:00:00.000000000',
                         str(from_time_in_days_since_1970(
                             to_time_in_days_since_1970('2018-06-08T12:00'))))
        self.assertEqual('2019-10-04T10:13:48.538000000',
                         str(from_time_in_days_since_1970(
                             to_time_in_days_since_1970('04-OCT-2019 10:13:48.538184'))))


class GetTimeRangeTest(unittest.TestCase):

    def test_get_time_range_from_data(self):
        cube = new_cube(drop_bounds=True)
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_with_irregular_data(self):
        cube = new_cube(drop_bounds=True,
                        time_freq='M')
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-31T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-06-30T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_with_irregular_data_and_no_metadata(self):
        cube = new_cube(drop_bounds=True,
                        time_freq='M')
        cube.attrs.pop('time_coverage_start')
        cube.attrs.pop('time_coverage_end')
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-02-14T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-06-14T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_cftime(self):
        cube = new_cube(drop_bounds=True,
                        use_cftime=True,
                        time_dtype=None)
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_with_irregular_cftime_data(self):
        cube = new_cube(drop_bounds=True,
                        time_freq='M',
                        use_cftime=True,
                        time_dtype=None)
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-31T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-06-30T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_with_irregular_cftime_data_and_no_metadata(self):
        cube = new_cube(drop_bounds=True,
                        time_freq='M',
                        use_cftime=True,
                        time_dtype=None)
        cube.attrs.pop('time_coverage_start')
        cube.attrs.pop('time_coverage_end')
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-02-14T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-06-14T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_time_named_t(self):
        cube = new_cube(drop_bounds=True, time_name='t')
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_additional_t_variable(self):
        import xarray as xr
        start_time_data = pd.date_range(start='2010-01-03T12:00:00',
                                        periods=5,
                                        freq='5D').values.astype(dtype='datetime64[s]')
        start_time = xr.DataArray(start_time_data, dims='time')
        end_time_data = pd.date_range(start='2010-01-07T12:00:00',
                                      periods=5,
                                      freq='5D').values.astype(dtype='datetime64[s]')
        end_time = xr.DataArray(end_time_data, dims='time')
        cube = new_cube(drop_bounds=True,
                        time_start='2010-01-05T12:00:00',
                        time_freq='5D',
                        variables=dict(start_time=start_time, end_time=end_time))
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-03T12:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-27T12:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_start_and_end_time_arrays(self):
        cube = new_cube(drop_bounds=True,
                        use_cftime=True,
                        time_dtype=None)
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_bounds(self):
        cube = new_cube()
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_attrs(self):
        cube = new_cube()
        time_range = get_time_range_from_attrs(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00', pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00', pd.Timestamp(time_range[1]).isoformat())

    def test_get_start_time_from_attrs(self):
        cube = new_cube()
        start_time = get_start_time_from_attrs(cube)
        self.assertEqual('2010-01-01T00:00:00', pd.Timestamp(start_time).isoformat())

    def test_get_end_time_from_attrs(self):
        cube = new_cube()
        end_time = get_end_time_from_attrs(cube)
        self.assertEqual('2010-01-06T00:00:00', pd.Timestamp(end_time).isoformat())


class TimestampToIsoStringTest(unittest.TestCase):
    def test_it_with_default_res(self):
        self.assertEqual("2018-09-05T00:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05")))
        self.assertEqual("2018-09-05T10:35:42Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05 10:35:42")))
        self.assertEqual("2018-09-05T10:35:42Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05 10:35:42.164")))
        self.assertEqual("2019-10-04T10:13:49Z",
                         timestamp_to_iso_string(pd.to_datetime("04-OCT-2019 10:13:48.538184")))

    def test_it_with_h_res(self):
        self.assertEqual("2018-09-05T00:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05"),
                                                 freq="H"))
        self.assertEqual("2018-09-05T11:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05 10:35:42"),
                                                 freq="H"))
        self.assertEqual("2018-09-05T11:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05 10:35:42.164"),
                                                 freq="H"))
        self.assertEqual("2019-10-04T10:00:00Z",
                         timestamp_to_iso_string(pd.to_datetime("04-OCT-2019 10:13:48.538184"),
                                                 freq="H"))
