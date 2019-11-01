import unittest

import numpy as np
import pandas as pd

from test.sampledata import create_highroc_dataset
from xcube.util.timecoord import add_time_coords, from_time_in_days_since_1970, timestamp_to_iso_string, \
    to_time_in_days_since_1970


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
                         str(from_time_in_days_since_1970(to_time_in_days_since_1970('201706071200'))))
        self.assertEqual('2017-06-08T12:00:00.000000000',
                         str(from_time_in_days_since_1970(to_time_in_days_since_1970('201706081200'))))
        self.assertEqual('2018-06-08T12:00:00.000000000',
                         str(from_time_in_days_since_1970(to_time_in_days_since_1970('2018-06-08 12:00'))))
        self.assertEqual('2018-06-08T12:00:00.000000000',
                         str(from_time_in_days_since_1970(to_time_in_days_since_1970('2018-06-08T12:00'))))
        self.assertEqual('2019-10-04T10:13:48.538000000',
                         str(from_time_in_days_since_1970(to_time_in_days_since_1970('04-OCT-2019 10:13:48.538184'))))


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
                         timestamp_to_iso_string(np.datetime64("2018-09-05"), freq="H"))
        self.assertEqual("2018-09-05T11:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05 10:35:42"), freq="H"))
        self.assertEqual("2018-09-05T11:00:00Z",
                         timestamp_to_iso_string(np.datetime64("2018-09-05 10:35:42.164"), freq="H"))
        self.assertEqual("2019-10-04T10:00:00Z",
                         timestamp_to_iso_string(pd.to_datetime("04-OCT-2019 10:13:48.538184"), freq="H"))
