import os
import unittest

from test.sampledata import new_test_cube, new_test_dataset
from xcube.api import open_dataset, read_dataset, write_dataset, dump_dataset, chunk_dataset

TEST_NC_FILE = "test.nc"


class OpenReadWriteDatasetTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        dataset = new_test_cube()
        dataset.to_netcdf(TEST_NC_FILE, mode="w")
        dataset.close()

    def tearDown(self):
        os.remove(TEST_NC_FILE)
        super().tearDown()

    def test_open_dataset(self):
        with open_dataset(TEST_NC_FILE) as ds:
            self.assertIsNotNone(ds)

    def test_read_dataset(self):
        ds = read_dataset(TEST_NC_FILE)
        self.assertIsNotNone(ds)
        ds.close()

    def test_write_dataset(self):
        TEST_NC_FILE_2 = "test-2.nc"

        dataset = new_test_cube()
        try:
            write_dataset(dataset, TEST_NC_FILE_2)
            self.assertTrue(os.path.isfile(TEST_NC_FILE_2))
        finally:
            if os.path.isfile(TEST_NC_FILE_2):
                os.remove(TEST_NC_FILE_2)


class ChunkDatasetTest(unittest.TestCase):
    def test_chunk_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=10, lon=20),
                                        format_name="zarr")
        self.assertEqual({'chunks': (1, 10, 20)}, chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunks': (1, 10, 20)}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=20, lon=40),
                                        format_name="netcdf4")
        self.assertEqual({'chunksizes': (1, 20, 40)}, chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunksizes': (1, 20, 40)}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=20, lon=40))
        self.assertEqual({}, chunked_dataset.precipitation.encoding)
        self.assertEqual({}, chunked_dataset.temperature.encoding)

    def test_unchunk_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        for var in dataset.data_vars.values():
            var.encoding.update({"chunks": (5, 180, 360), "_FillValue": -999.0})

        chunked_dataset = chunk_dataset(dataset, format_name="zarr")
        self.assertEqual({"_FillValue": -999.0}, chunked_dataset.precipitation.encoding)
        self.assertEqual({"_FillValue": -999.0}, chunked_dataset.temperature.encoding)


class DumpDatasetTest(unittest.TestCase):
    def test_dump_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)
        for var in dataset.variables.values():
            var.encoding.update({"_FillValue": 999.0})

        text = dump_dataset(dataset)
        self.assertIn("<xarray.Dataset>", text)
        self.assertIn("Dimensions:        (lat: 180, lon: 360, time: 5)\n", text)
        self.assertIn("Coordinates:\n", text)
        self.assertIn("  * lon            (lon) float64 ", text)
        self.assertIn("Data variables:\n", text)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", text)
        self.assertNotIn("Encoding for coordinate variable 'lat':\n", text)
        self.assertNotIn("Encoding for data variable 'temperature':\n", text)
        self.assertNotIn("    _FillValue:  999.0\n", text)

        text = dump_dataset(dataset, show_var_encoding=True)
        self.assertIn("<xarray.Dataset>", text)
        self.assertIn("Dimensions:        (lat: 180, lon: 360, time: 5)\n", text)
        self.assertIn("Coordinates:\n", text)
        self.assertIn("  * lon            (lon) float64 ", text)
        self.assertIn("Data variables:\n", text)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", text)
        self.assertIn("Encoding for coordinate variable 'lat':\n", text)
        self.assertIn("Encoding for data variable 'temperature':\n", text)
        self.assertIn("    _FillValue:  999.0\n", text)

        text = dump_dataset(dataset, ["precipitation"])
        self.assertIn("<xarray.DataArray 'precipitation' (time: 5, lat: 180, lon: 360)>\n", text)
        self.assertNotIn("Encoding:\n", text)
        self.assertNotIn("    _FillValue:  999.0", text)

        text = dump_dataset(dataset, ["precipitation"], show_var_encoding=True)
        self.assertIn("<xarray.DataArray 'precipitation' (time: 5, lat: 180, lon: 360)>\n", text)
        self.assertIn("Encoding:\n", text)
        self.assertIn("    _FillValue:  999.0", text)
