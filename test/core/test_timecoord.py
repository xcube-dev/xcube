import unittest

import numpy as np
import pandas as pd
import pytest

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


# from xcube.core.timecoord import find_datetime_format
# from xcube.core.timecoord import get_timestamp_from_string
# from xcube.core.timecoord import get_timestamps_from_string


class AddTimeCoordsTest(unittest.TestCase):

    def test_add_time_coords_point(self):
        dataset = create_highroc_dataset()
        dataset_with_time = add_time_coords(dataset,
                                            (365 * 47 + 20, 365 * 47 + 20))
        self.assertIsNot(dataset_with_time, dataset)
        self.assertIn('time', dataset_with_time)
        self.assertEqual(dataset_with_time.time.shape, (1,))
        self.assertNotIn('time_bnds', dataset_with_time)

    def test_add_time_coords_range(self):
        dataset = create_highroc_dataset()
        dataset_with_time = add_time_coords(dataset,
                                            (365 * 47 + 20, 365 * 47 + 21))
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
                         to_time_in_days_since_1970(
                             '04-OCT-2019 10:13:48.538184'))

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
                             to_time_in_days_since_1970(
                                 '04-OCT-2019 10:13:48.538184'))))


class GetTimeRangeTest(unittest.TestCase):

    def test_get_time_range_from_data(self):
        cube = new_cube(drop_bounds=True)
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_with_irregular_data(self):
        cube = new_cube(drop_bounds=True,
                        time_freq='M')
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-31T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-06-30T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_with_irregular_data_and_no_metadata(self):
        cube = new_cube(drop_bounds=True,
                        time_freq='M')
        cube.attrs.pop('time_coverage_start')
        cube.attrs.pop('time_coverage_end')
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-02-14T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-06-14T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_cftime(self):
        cube = new_cube(drop_bounds=True,
                        use_cftime=True,
                        time_dtype=None)
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_with_irregular_cftime_data(self):
        cube = new_cube(drop_bounds=True,
                        time_freq='M',
                        use_cftime=True,
                        time_dtype=None)
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-31T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-06-30T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_with_irregular_cftime_data_and_no_metadata(
            self):
        cube = new_cube(drop_bounds=True,
                        time_freq='M',
                        use_cftime=True,
                        time_dtype=None)
        cube.attrs.pop('time_coverage_start')
        cube.attrs.pop('time_coverage_end')
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-02-14T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-06-14T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_time_named_t(self):
        cube = new_cube(drop_bounds=True, time_name='t')
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_additional_t_variable(self):
        import xarray as xr
        start_time_data = pd.date_range(start='2010-01-03T12:00:00',
                                        periods=5,
                                        freq='5D').values.astype(
            dtype='datetime64[s]')
        start_time = xr.DataArray(start_time_data, dims='time')
        end_time_data = pd.date_range(start='2010-01-07T12:00:00',
                                      periods=5,
                                      freq='5D').values.astype(
            dtype='datetime64[s]')
        end_time = xr.DataArray(end_time_data, dims='time')
        cube = new_cube(drop_bounds=True,
                        time_start='2010-01-05T12:00:00',
                        time_freq='5D',
                        variables=dict(start_time=start_time,
                                       end_time=end_time))
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-03T12:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-27T12:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_start_and_end_time_arrays(self):
        cube = new_cube(drop_bounds=True,
                        use_cftime=True,
                        time_dtype=None)
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_data_bounds(self):
        cube = new_cube()
        time_range = get_time_range_from_data(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_time_range_from_attrs(self):
        cube = new_cube()
        time_range = get_time_range_from_attrs(cube)
        self.assertIsNotNone(time_range)
        self.assertEqual('2010-01-01T00:00:00',
                         pd.Timestamp(time_range[0]).isoformat())
        self.assertEqual('2010-01-06T00:00:00',
                         pd.Timestamp(time_range[1]).isoformat())

    def test_get_start_time_from_attrs(self):
        cube = new_cube()
        start_time = get_start_time_from_attrs(cube)
        self.assertEqual('2010-01-01T00:00:00',
                         pd.Timestamp(start_time).isoformat())

    def test_get_end_time_from_attrs(self):
        cube = new_cube()
        end_time = get_end_time_from_attrs(cube)
        self.assertEqual('2010-01-06T00:00:00',
                         pd.Timestamp(end_time).isoformat())


class TimestampToIsoStringTest(unittest.TestCase):
    def test_it_with_default_res(self):
        self.assertEqual("2018-09-05T00:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05")))
        self.assertEqual("2018-09-05T10:35:42Z",
                         timestamp_to_iso_string(
                             np.datetime64("2018-09-05 10:35:42")))
        self.assertEqual("2018-09-05T10:35:42Z",
                         timestamp_to_iso_string(
                             np.datetime64("2018-09-05 10:35:42.164")))
        self.assertEqual("2019-10-04T10:13:49Z",
                         timestamp_to_iso_string(
                             pd.to_datetime("04-OCT-2019 10:13:48.538184")))

    def test_it_with_ceil_round_fn(self):
        self.assertEqual("2018-09-05T00:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05"),
                                                 round_fn="ceil"))
        self.assertEqual("2018-09-05T10:35:42Z",
                         timestamp_to_iso_string(
                             np.datetime64("2018-09-05 10:35:42"),
                             round_fn="ceil"))
        self.assertEqual("2018-09-05T10:35:43Z",
                         timestamp_to_iso_string(
                             np.datetime64("2018-09-05 10:35:42.164"),
                             round_fn="ceil"))
        self.assertEqual("2019-10-04T10:13:49Z",
                         timestamp_to_iso_string(
                             pd.to_datetime("04-OCT-2019 10:13:48.538184"),
                             round_fn="ceil"))

    def test_it_with_floor_round_fn(self):
        self.assertEqual("2018-09-05T00:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05"),
                                                 round_fn="floor"))
        self.assertEqual("2018-09-05T10:35:42Z",
                         timestamp_to_iso_string(
                             np.datetime64("2018-09-05 10:35:42"),
                             round_fn="floor"))
        self.assertEqual("2018-09-05T10:35:42Z",
                         timestamp_to_iso_string(
                             np.datetime64("2018-09-05 10:35:42.164"),
                             round_fn="floor"))
        self.assertEqual("2019-10-04T10:13:48Z",
                         timestamp_to_iso_string(
                             pd.to_datetime("04-OCT-2019 10:13:48.538184"),
                             round_fn="floor"))

    def test_it_with_array_round_fn(self):
        var = [np.datetime64("2018-09-05 10:35:42.564"),
               np.datetime64("2018-09-06 10:35:42.564"),
               np.datetime64("2018-09-07 10:35:42.564"),
               pd.to_datetime("04-OCT-2019 10:13:48.038184")
               ]
        expected_values = ["2018-09-05T10:35:42Z",
                           "2018-09-06T10:35:43Z",
                           "2018-09-07T10:35:43Z",
                           "2019-10-04T10:13:49Z"]
        values = [timestamp_to_iso_string(var[0], round_fn="floor")] +\
                 list(map(timestamp_to_iso_string, var[1:-1])) +\
                 [timestamp_to_iso_string(var[-1], round_fn="ceil")]
        self.assertEqual(expected_values, values)


    # noinspection PyMethodMayBeStatic
    def test_it_with_invalid_round_fn(self):
        with pytest.raises(ValueError,
                           match=r"round_fn must be one of"
                                 r" \('ceil', 'floor', 'round'\)"):
            timestamp_to_iso_string(np.datetime64("2018-09-05 10:35:42.164"),
                                    round_fn="foo")

    def test_it_with_h_res(self):
        self.assertEqual("2018-09-05T00:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05"),
                                                 freq="H"))
        self.assertEqual("2018-09-05T11:00:00Z",
                         timestamp_to_iso_string(
                             np.datetime64("2018-09-05 10:35:42"),
                             freq="H"))
        self.assertEqual("2018-09-05T11:00:00Z",
                         timestamp_to_iso_string(
                             np.datetime64("2018-09-05 10:35:42.164"),
                             freq="H"))
        self.assertEqual("2019-10-04T10:00:00Z",
                         timestamp_to_iso_string(
                             pd.to_datetime("04-OCT-2019 10:13:48.538184"),
                             freq="H"))

# class TimeStampsTest(unittest.TestCase):
#
#     def test_find_datetime_format(self):
#         dt_format, start_index, end_index = find_datetime_format('ftze20140305131415dgs')
#         self.assertEqual('%Y%m%d%H%M%S', dt_format)
#         self.assertEqual(4, start_index)
#         self.assertEqual(18, end_index)
#
#         dt_format, start_index, end_index = find_datetime_format('ftze201403051314dgs')
#         self.assertEqual('%Y%m%d%H%M', dt_format)
#         self.assertEqual(4, start_index)
#         self.assertEqual(16, end_index)
#
#         dt_format, start_index, end_index = find_datetime_format('ft2ze20140307dgs')
#         self.assertEqual('%Y%m%d', dt_format)
#         self.assertEqual(5, start_index)
#         self.assertEqual(13, end_index)
#
#         dt_format, start_index, end_index = find_datetime_format('ft2ze201512dgs')
#         self.assertEqual('%Y%m', dt_format)
#         self.assertEqual(5, start_index)
#         self.assertEqual(11, end_index)
#
#         dt_format, start_index, end_index = find_datetime_format('ft2s6ze2016dgs')
#         self.assertEqual('%Y', dt_format)
#         self.assertEqual(7, start_index)
#         self.assertEqual(11, end_index)
#
#     def test_get_timestamp_from_string(self):
#         timestamp = get_timestamp_from_string('ftze20140305131415dg0023s')
#         self.assertEqual(pd.Timestamp('2014-03-05T13:14:15'), timestamp)
#
#         timestamp = get_timestamp_from_string('ftze201403051314dgs')
#         self.assertEqual(pd.Timestamp('2014-03-05T13:14:00'), timestamp)
#
#         timestamp = get_timestamp_from_string('ftze20140305dgs')
#         self.assertEqual(pd.Timestamp('2014-03-05T00:00:00'), timestamp)
#
#         timestamp = get_timestamp_from_string('ftze201403dgs')
#         self.assertEqual(pd.Timestamp('2014-03-01T00:00:00'), timestamp)
#
#         timestamp = get_timestamp_from_string('ftze2014dgs')
#         self.assertEqual(pd.Timestamp('2014-01-01T00:00:00'), timestamp)
#
#     def test_get_timestamps_from_string(self):
#         timestamp_1, timestamp_2 = get_timestamps_from_string(
#             '20020401-20020406-ESACCI-L3C_AEROSOL-AEX-GOMOS_ENVISAT-AERGOM_5days-fv2.19.nc')
#         self.assertEqual(pd.Timestamp('2002-04-01'), timestamp_1)
#         self.assertEqual(pd.Timestamp('2002-04-06'), timestamp_2)
#
#         timestamp_1, timestamp_2 = get_timestamps_from_string(
#             '20020401-ESACCI-L3C_AEROSOL-AEX-GOMOS_ENVISAT-AERGOM_5days-fv2.19.nc')
#         self.assertEqual(pd.Timestamp('2002-04-01'), timestamp_1)
#         self.assertIsNone(timestamp_2)
