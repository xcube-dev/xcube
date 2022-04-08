import os

import xarray as xr

from test.cli.helpers import CliDataTest, TEST_NC_FILE, TEST_ZARR_DIR


class ChunkTest(CliDataTest):
    TEST_OUTPUT = "test-chunked.zarr"

    def outputs(self):
        return [self.TEST_OUTPUT]

    def test_chunk_zarr(self):
        output_path = ChunkTest.TEST_OUTPUT
        result = self.invoke_cli(["chunk",
                                  TEST_ZARR_DIR,
                                  "-o", output_path,
                                  "--chunks", "time=1,lat=20,lon=40"])
        self.assertEqual("", result.output)
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(output_path))

        ds = xr.open_zarr(output_path)
        self.assertIn("precipitation", ds)
        precipitation = ds["precipitation"]
        self.assertTrue(hasattr(precipitation, "encoding"))
        self.assertIn("chunks", precipitation.encoding)
        self.assertEqual(precipitation.encoding["chunks"], (1, 20, 40))

    # TODO (forman): this test fails
    # netCDF4\_netCDF4.pyx:2437: in netCDF4._netCDF4.Dataset.createVariable
    # ValueError: cannot specify chunksizes for a contiguous dataset
    #
    # def test_chunk_nc(self):
    #     output_path = "test-chunked.nc"
    #     result = self.invoke_cli(["chunk",
    #                               TEST_NC_FILE,
    #                               "-o", output_path,
    #                               "--chunks", "time=1,lat=20,lon=40"])
    #     self.assertEqual("", result.output)
    #     self.assertEqual(0, result.exit_code)
    #     self.assertTrue(os.path.isdir(output_path))
    #     try:
    #         ds = xr.open_zarr(output_path)
    #         self.assertIn("precipitation", ds)
    #         precipitation = ds["precipitation"]
    #         self.assertTrue(hasattr(precipitation, "encoding"))
    #         self.assertIn("chunksizes", precipitation.encoding)
    #         self.assertEqual(precipitation.encoding["chunksizes"], (1, 20, 40))
    #     finally:
    #         os.remove(output_path)

    def test_chunk_size_syntax(self):
        result = self.invoke_cli(["chunk",
                                  TEST_NC_FILE,
                                  "-o", "test-chunked.zarr",
                                  "--chunks", "time=1,lat!gnnn,lon=40"])
        self.assertEqual("Error: Invalid value for CHUNKS:"
                         " 'time=1,lat!gnnn,lon=40'\n",
                         result.stderr)
        self.assertEqual(1, result.exit_code)

    def test_chunk_size_not_an_int(self):
        result = self.invoke_cli(["chunk",
                                  TEST_NC_FILE,
                                  "-o", "test-chunked.zarr",
                                  "--chunks", "time=1,lat=20.3,lon=40"])
        self.assertEqual("Error: Invalid value for CHUNKS,"
                         " chunk sizes must be positive integers:"
                         " time=1,lat=20.3,lon=40\n",
                         result.stderr)
        self.assertEqual(1, result.exit_code)

    def test_chunk_size_not_a_dim(self):
        result = self.invoke_cli(["chunk",
                                  TEST_NC_FILE,
                                  "-o", "test-chunked.zarr",
                                  "--chunks", "time=1,lati=20,lon=40"])
        self.assertEqual("Error: Invalid value for CHUNKS,"
                         " 'lati' is not the name of any dimension:"
                         " time=1,lati=20,lon=40\n",
                         result.stderr)
        self.assertEqual(1, result.exit_code)
