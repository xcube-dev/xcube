# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from test.cli.helpers import CliDataTest, TEST_NC_FILE
import xarray as xr


class DumpTest(CliDataTest):
    def test_dump_ds(self):
        self.maxDiff = None
        with xr.set_options(display_width=80):
            result = self.invoke_cli(["dump", TEST_NC_FILE])

        # Use a regex to accommodate the differing output formats produced by
        # xarray 2023.8.0 and 2024.2.0.
        output_regex = r"""<xarray.Dataset>( Size: 8MB)?
Dimensions:        \(lon: 360, lat: 180, time: 5, bnds: 2\)
Coordinates:
  \* lon            \(lon\) float64 (3kB )?-179\.5 -178\.5 -177\.5 \.\.\. 177\.5 178\.5 179\.5
  \* lat            \(lat\) float64 (1kB )?-89\.5 -88\.5 -87\.5 -86\.5 \.\.\. (86\.5 )?87\.5 88\.5 89\.5
  \* time           \(time\) datetime64\[ns\] (40B )?2010-01-01T12:00:00 \.\.\. 2010-01-(05T1)?\.\.\.
    lon_bnds       \(lon, bnds\) float64 (6kB )?\.\.\.
    lat_bnds       \(lat, bnds\) float64 (3kB )?\.\.\.
    time_bnds      \(time, bnds\) datetime64\[ns\] (80B )?\.\.\.
Dimensions without coordinates: bnds
Data variables:
    precipitation  \(time, lat, lon\) float64 (3MB )?...
    temperature    \(time, lat, lon\) float64 (3MB )?...
    soil_moisture  \(time, lat, lon\) float64 (3MB )?...
Attributes:
    Conventions:           CF-1.7
    title:                 Test Cube
    time_coverage_start:   2010-01-01T00:00:00.000000000
    time_coverage_end:     2010-01-06T00:00:00.000000000
    geospatial_lon_min:    -180.0
    geospatial_lon_max:    180.0
    geospatial_lon_units:  degrees_east
    geospatial_lat_min:    -90.0
    geospatial_lat_max:    90.0
    geospatial_lat_units:  degrees_north
"""

        self.assertRegex(result.output, output_regex)
        self.assertEqual(0, result.exit_code)
