import os

import xarray as xr

from test.cli.helpers import CliDataTest, TEST_NC_FILE, TEST_ZARR_DIR


class DumpTest(CliDataTest):

    def test_dump_ds(self):
        result = self.invoke_cli(["dump", TEST_NC_FILE])
        self.assertEqual((
            "<xarray.Dataset>\n"
            "Dimensions:        (bnds: 2, lat: 180, lon: 360, time: 5)\n"
            "Coordinates:\n"
            "  * lon            (lon) float64 -179.5 -178.5 -177.5 ... 177.5 178.5 179.5\n"
            "  * lat            (lat) float64 -89.5 -88.5 -87.5 -86.5 ... 86.5 87.5 88.5 89.5\n"
            "  * time           (time) datetime64[ns] 2010-01-01T12:00:00 ... 2010-01-05T12:00:00\n"
            "    lon_bnds       (lon, bnds) float64 ...\n"
            "    lat_bnds       (lat, bnds) float64 ...\n"
            "    time_bnds      (time, bnds) datetime64[ns] ...\n"
            "Dimensions without coordinates: bnds\n"
            "Data variables:\n"
            "    precipitation  (time, lat, lon) float64 ...\n"
            "    temperature    (time, lat, lon) float64 ...\n"
            "    soil_moisture  (time, lat, lon) float64 ...\n"
            "Attributes:\n"
            "    Conventions:           CF-1.7\n"
            "    title:                 Test Cube\n"
            "    time_coverage_start:   2010-01-01 00:00:00\n"
            "    time_coverage_end:     2010-01-06 00:00:00\n"
            "    geospatial_lon_min:    -180.0\n"
            "    geospatial_lon_max:    180.0\n"
            "    geospatial_lon_units:  degrees_east\n"
            "    geospatial_lat_min:    -90.0\n"
            "    geospatial_lat_max:    90.0\n"
            "    geospatial_lat_units:  degrees_north\n"
        ), result.output)
        self.assertEqual(0, result.exit_code)


class Vars2DimTest(CliDataTest):
    TEST_OUTPUT = "test-vars2dim.zarr"

    def outputs(self):
        return [self.TEST_OUTPUT]

    def test_vars2dim(self):
        result = self.invoke_cli(["vars2dim", TEST_ZARR_DIR])

        output_path = self.TEST_OUTPUT
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(output_path))

        ds = xr.open_zarr(output_path)
        self.assertIn("var", ds.dims)
        self.assertEqual(3, ds.dims["var"])
        self.assertIn("var", ds.coords)
        self.assertIn("data", ds.data_vars)
        var_names = ds["var"]
        self.assertEqual(("var",), var_names.dims)
        self.assertTrue(hasattr(var_names, "encoding"))
        self.assertEqual(3, len(var_names))
        self.assertIn("precipitation", str(var_names[0]))
        self.assertIn("soil_moisture", str(var_names[1]))
        self.assertIn("temperature", str(var_names[2]))


class ChunkTest(CliDataTest):
    TEST_OUTPUT = "test-chunked.zarr"

    def outputs(self):
        return [self.TEST_OUTPUT]

    def test_chunk_zarr(self):
        output_path = ChunkTest.TEST_OUTPUT
        result = self.invoke_cli(["chunk",
                                  TEST_ZARR_DIR,
                                  output_path,
                                  "-c", "time=1,lat=20,lon=40"])
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
    #                               output_path,
    #                               "-c", "time=1,lat=20,lon=40"])
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


class LevelTest(CliDataTest):
    TEST_OUTPUT = "test.levels"

    def outputs(self):
        return [LevelTest.TEST_OUTPUT]

    def test_level_with_nc(self):
        result = self.invoke_cli(["level",
                                  "-t", "45",
                                  "-o", LevelTest.TEST_OUTPUT,
                                  TEST_NC_FILE,
                                  ])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelTest.TEST_OUTPUT))
        self.assertEqual({'0.zarr', '1.zarr', '2.zarr'}, set(os.listdir(LevelTest.TEST_OUTPUT)))

    def test_level_with_zarr(self):
        result = self.invoke_cli(["level",
                                  "-t", "45",
                                  "-o", LevelTest.TEST_OUTPUT,
                                  TEST_ZARR_DIR,
                                  ])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelTest.TEST_OUTPUT))
        self.assertEqual({'0.zarr', '1.zarr', '2.zarr'}, set(os.listdir(LevelTest.TEST_OUTPUT)))

    def test_level_with_zarr_link(self):
        result = self.invoke_cli(["level",
                                  "-l",
                                  "-t", "45",
                                  "-o", LevelTest.TEST_OUTPUT,
                                  TEST_ZARR_DIR,
                                  ])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelTest.TEST_OUTPUT))
        self.assertEqual({'0.link', '1.zarr', '2.zarr'}, set(os.listdir(LevelTest.TEST_OUTPUT)))

    def test_level_with_zarr_num_levels_max(self):
        result = self.invoke_cli(["level",
                                  "-t", "45",
                                  "-n", "2",
                                  "-o", LevelTest.TEST_OUTPUT,
                                  TEST_ZARR_DIR,
                                  ])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelTest.TEST_OUTPUT))
        self.assertEqual({'0.zarr', '1.zarr'}, set(os.listdir(LevelTest.TEST_OUTPUT)))

    def test_invalid_inputs(self):
        result = self.invoke_cli(["level",
                                  "-t", "a45",
                                  "-o", LevelTest.TEST_OUTPUT,
                                  TEST_NC_FILE,
                                  ])
        self.assertEqual("Error: Expected a pair of positive integers <tile-width>,<tile-height>\n",
                         result.output)
        self.assertEqual(1, result.exit_code)

        result = self.invoke_cli(["level",
                                  "-t", "-3",
                                  "-o", LevelTest.TEST_OUTPUT,
                                  TEST_NC_FILE,
                                  ])
        self.assertEqual("Error: <tile-size> must comprise positive integers\n",
                         result.output)
        self.assertEqual(1, result.exit_code)

        result = self.invoke_cli(["level",
                                  "-t", "45,45,45",
                                  "-o", LevelTest.TEST_OUTPUT,
                                  TEST_NC_FILE])
        self.assertEqual("Error: Expected a pair of positive integers <tile-width>,<tile-height>\n",
                         result.output)
        self.assertEqual(1, result.exit_code)

        result = self.invoke_cli(["level",
                                  "-n", "0",
                                  "-o", LevelTest.TEST_OUTPUT,
                                  TEST_NC_FILE])
        self.assertEqual("Error: <num-levels-max> must be a positive integer\n",
                         result.output)
        self.assertEqual(1, result.exit_code)
