from test.cli.helpers import CliDataTest, TEST_NC_FILE


class DumpTest(CliDataTest):

    def test_dump_ds(self):
        self.maxDiff = None
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
            "    time_coverage_start:   2010-01-01T00:00:00\n"
            "    time_coverage_end:     2010-01-06T00:00:00\n"
            "    geospatial_lon_min:    -180.0\n"
            "    geospatial_lon_max:    180.0\n"
            "    geospatial_lon_units:  degrees_east\n"
            "    geospatial_lat_min:    -90.0\n"
            "    geospatial_lat_max:    90.0\n"
            "    geospatial_lat_units:  degrees_north\n"
        ), result.output)
        self.assertEqual(0, result.exit_code)
