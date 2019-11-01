import os
import unittest

import numpy as np

from xcube.core.new import new_cube
from xcube.core.readwrite import open_dataset, read_dataset, write_dataset

TEST_NC_FILE = "test.nc"
TEST_NC_FILE_2 = "test-2.nc"


class OpenReadWriteDatasetTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = new_cube(variables=dict(precipitation=0.2, temperature=279.1))
        self.dataset.to_netcdf(TEST_NC_FILE, mode="w")
        self.dataset.close()

    def tearDown(self):
        self.dataset = None
        os.remove(TEST_NC_FILE)
        super().tearDown()

    def test_open_dataset(self):
        with open_dataset(TEST_NC_FILE) as ds:
            self.assertIsNotNone(ds)
            np.testing.assert_array_equal(ds.time.values, self.dataset.time.values)
            np.testing.assert_array_equal(ds.lat.values, self.dataset.lat.values)
            np.testing.assert_array_equal(ds.lon.values, self.dataset.lon.values)

    def test_read_dataset(self):
        ds = read_dataset(TEST_NC_FILE)
        self.assertIsNotNone(ds)
        np.testing.assert_array_equal(ds.time.values, self.dataset.time.values)
        np.testing.assert_array_equal(ds.lat.values, self.dataset.lat.values)
        np.testing.assert_array_equal(ds.lon.values, self.dataset.lon.values)
        ds.close()

    def test_write_dataset(self):

        dataset = new_cube()
        try:
            write_dataset(dataset, TEST_NC_FILE_2)
            self.assertTrue(os.path.isfile(TEST_NC_FILE_2))
        finally:
            if os.path.isfile(TEST_NC_FILE_2):
                os.remove(TEST_NC_FILE_2)
