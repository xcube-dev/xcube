import os
import shutil
import unittest
from abc import ABCMeta
from typing import List

import click
import click.testing
import numpy as np
import pandas as pd
import xarray as xr

from xcube.cli import cli, _parse_kwargs

TEST_NC_FILE = "test.nc"
TEST_ZARR_DIR = "test.zarr"


class CliTest(unittest.TestCase, metaclass=ABCMeta):

    def invoke_cli(self, args: List[str]):
        self.runner = click.testing.CliRunner()
        return self.runner.invoke(cli, args, catch_exceptions=False)

    def setUp(self):
        super().setUp()
        dataset = create_test_dataset()
        dataset.to_netcdf(TEST_NC_FILE, mode="w")
        dataset.to_zarr(TEST_ZARR_DIR, mode="w")

    def tearDown(self):
        if os.path.isdir(TEST_ZARR_DIR):
            shutil.rmtree(TEST_ZARR_DIR, ignore_errors=True)
        os.remove(TEST_NC_FILE)
        super().tearDown()


class DumpTest(CliTest):

    def test_dump_ds(self):
        result = self.invoke_cli(["dump", TEST_NC_FILE])
        self.assertEqual((
            "<xarray.Dataset>\n"
            "Dimensions:        (lat: 100, lon: 200, time: 5)\n"
            "Coordinates:\n"
            "  * time           (time) datetime64[ns] 2010-01-01 2010-01-02 2010-01-03 ...\n"
            "  * lat            (lat) float64 50.0 50.02 50.04 50.06 50.08 50.1 50.12 ...\n"
            "  * lon            (lon) float64 0.0 0.0201 0.0402 0.0603 0.0804 0.1005 ...\n"
            "Data variables:\n"
            "    precipitation  (time, lat, lon) float64 ...\n"
            "    temperature    (time, lat, lon) float64 ...\n"
            "Attributes:\n"
            "    time_coverage_start:  2010-01-01 00:00:00\n"
            "    time_coverage_end:    2010-01-05 00:00:00\n"
        ), result.output)
        self.assertEqual(0, result.exit_code)


class ChunkTest(CliTest):

    def test_chunk_zarr(self):
        output_path = "test-chunked.zarr"
        result = self.invoke_cli(["chunk",
                                  TEST_ZARR_DIR,
                                  output_path,
                                  "-c", "time=1,lat=20,lon=40"])
        self.assertEqual("", result.output)
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(output_path))
        try:
            ds = xr.open_zarr(output_path)
            self.assertIn("precipitation", ds)
            precipitation = ds["precipitation"]
            self.assertTrue(hasattr(precipitation, "encoding"))
            self.assertIn("chunks", precipitation.encoding)
            self.assertEqual(precipitation.encoding["chunks"], (1, 20, 40))
        finally:
            shutil.rmtree(output_path, ignore_errors=True)

    def test_chunk_nc(self):
        output_path = "test-chunked.nc"
        result = self.invoke_cli(["chunk",
                                  TEST_NC_FILE,
                                  output_path,
                                  "-c", "time=1,lat=20,lon=40"])
        self.assertEqual("", result.output)
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(output_path))
        try:
            ds = xr.open_zarr(output_path)
            self.assertIn("precipitation", ds)
            precipitation = ds["precipitation"]
            self.assertTrue(hasattr(precipitation, "encoding"))
            self.assertIn("chunksizes", precipitation.encoding)
            self.assertEqual(precipitation.encoding["chunksizes"], (1, 20, 40))
        finally:
            os.remove(output_path)

    def test_chunk_size_syntax(self):
        result = self.invoke_cli(["chunk",
                                  TEST_NC_FILE,
                                  "test-chunked.zarr",
                                  "-c", "time=1,lat!gnnn,lon=40"])
        self.assertEqual("Error: Invalid value for <chunks>:"
                         " 'time=1,lat!gnnn,lon=40'\n",
                         result.output)
        self.assertEqual(1, result.exit_code)

    def test_chunk_size_not_an_int(self):
        result = self.invoke_cli(["chunk",
                                  TEST_NC_FILE,
                                  "test-chunked.zarr",
                                  "-c", "time=1,lat=20.3,lon=40"])
        self.assertEqual("Error: Invalid value for <chunks>,"
                         " chunk sizes must be positive integers:"
                         " time=1,lat=20.3,lon=40\n",
                         result.output)
        self.assertEqual(1, result.exit_code)

    def test_chunk_size_not_a_dim(self):
        result = self.invoke_cli(["chunk",
                                  TEST_NC_FILE,
                                  "test-chunked.zarr",
                                  "-c", "time=1,lati=20,lon=40"])
        self.assertEqual("Error: Invalid value for <chunks>,"
                         " 'lati' is not the name of any dimension:"
                         " time=1,lati=20,lon=40\n",
                         result.output)
        self.assertEqual(1, result.exit_code)


class ParseTest(unittest.TestCase):

    def test_parse_kwargs(self):
        self.assertEqual(dict(),
                         _parse_kwargs("", metavar="<chunks>"))
        self.assertEqual(dict(time=1, lat=256, lon=512),
                         _parse_kwargs("time=1, lat=256, lon=512", metavar="<chunks>"))

        with self.assertRaises(click.ClickException) as cm:
            _parse_kwargs("45 * 'A'", metavar="<chunks>")
        self.assertEqual("Invalid value for <chunks>: \"45 * 'A'\"",
                         f"{cm.exception}")

        with self.assertRaises(click.ClickException) as cm:
            _parse_kwargs("9==2")
        self.assertEqual("Invalid value: '9==2'",
                         f"{cm.exception}")


def create_test_dataset(size=100, periods=5):
    dims = ("time", "lat", "lon")
    w = 2 * size
    h = size
    time = pd.date_range('2010-01-01', periods=periods)
    lon = np.linspace(0, 4, w)
    lat = np.linspace(50, 52, h)
    precipitation_var = xr.DataArray(np.random.rand(periods, h, w), coords=(time, lat, lon), dims=dims)
    temperature_var = xr.DataArray(np.random.rand(periods, h, w), coords=(time, lat, lon), dims=dims)
    ds = xr.Dataset({"precipitation": precipitation_var, "temperature": temperature_var})
    ds.attrs.update(dict(time_coverage_start=str(time[0]), time_coverage_end=str(time[-1])))
    return ds
