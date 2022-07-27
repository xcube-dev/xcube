from test.cli.helpers import CliDataTest, TEST_NC_FILE
import xarray as xr


class DumpTest(CliDataTest):

    def test_dump_ds(self):
        self.maxDiff = None
        with xr.set_options(display_width=80):
            result = self.invoke_cli(["dump", TEST_NC_FILE])

        # xarray v0.16.0 changed line truncation behaviour (see
        # https://github.com/pydata/xarray/issues/3759 ), so this test checks
        # the output against two expected values to cover both pre- and
        # post-change behaviour.
        output_template = (
            "<xarray.Dataset>\n"
            "Dimensions:        (time: 5, lat: 180, lon: 360, bnds: 2)\n"
            "Coordinates:\n"
            "  * lon            (lon) float64 -179.5 -178.5 -177.5 ... 177.5 178.5 179.5\n"
            "  * lat            (lat) float64 -89.5 -88.5 -87.5 -86.5 ... 86.5 87.5 88.5 89.5\n"
            "  * time           (time) datetime64[ns] 2010-01-01T12:00:00 ... 2010-01-05T1%s\n"
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
        )
        # xarray v2022.06.0 changed dimension order (see
        # https://github.com/pydata/xarray/pull/6601 and
        # https://github.com/pydata/xarray/issues/6600)
        output_template_1 = (
            "<xarray.Dataset>\n"
            "Dimensions:        (lon: 360, lat: 180, time: 5, bnds: 2)\n"
            "Coordinates:\n"
            "  * lon            (lon) float64 -179.5 -178.5 -177.5 ... 177.5 178.5 179.5\n"
            "  * lat            (lat) float64 -89.5 -88.5 -87.5 -86.5 ... 86.5 87.5 88.5 89.5\n"
            "  * time           (time) datetime64[ns] 2010-01-01T12:00:00 ... 2010-01-05T1%s\n"
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
        )

        possible_outputs = [
            output_template % '2:00:00',  # for xarray v0.15.1 and below
            output_template % '...',  # for xarray v0.16.0 and above
            output_template_1 % '...'  # for xarray v2022.06.0 and above
        ]
        print(result.output)
        self.assertIn(result.output, possible_outputs)
        self.assertEqual(0, result.exit_code)
