import numpy as np
import xarray as xr

from test.cli.helpers import CliTest
from xcube.core.dsio import rimraf
from xcube.core.dsio import write_cube
from xcube.core.new import new_cube


class VerifyTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['verify', '--help'])
        self.assertEqual(0, result.exit_code)


class VerifyDataTest(CliTest):
    TEST_CUBE = "test.zarr"

    def setUp(self) -> None:
        rimraf(self.TEST_CUBE)

    def tearDown(self) -> None:
        rimraf(self.TEST_CUBE)

    def test_verify_ok(self):
        cube = new_cube(variables=dict(precipitation=0.5))
        write_cube(cube, self.TEST_CUBE, "zarr", cube_asserted=True)
        result = self.invoke_cli(['verify', self.TEST_CUBE])
        self.assertEqual(0, result.exit_code)
        self.assertEqual("INPUT is a valid cube.\n",
                         result.stdout)

    def test_verify_failure(self):
        cube = new_cube(variables=dict(precipitation=0.5))
        cube["chl"] = xr.DataArray(np.random.rand(cube.dims["lat"], cube.dims["lon"]),
                                   dims=("lat", "lon"),
                                   coords=dict(lat=cube.lat, lon=cube.lon))
        write_cube(cube, self.TEST_CUBE, "zarr", cube_asserted=True)

        result = self.invoke_cli(['verify', self.TEST_CUBE])
        self.assertEqual(3, result.exit_code)
        self.assertEqual("INPUT is not a valid cube due to the following reasons:\n"
                         "- dimensions of data variable 'chl' must be ('time', ..., 'lat', 'lon'),"
                         " but were ('lat', 'lon') for 'chl'\n"
                         "- dimensions of all data variables must be same, but found ('lat', 'lon')"
                         " for 'chl' and ('time', 'lat', 'lon') for 'precipitation'\n"
                         "- all data variables must have same chunk sizes, but found ((90, 90), (360,))"
                         " for 'chl' and ((3, 2), (90, 90), (180, 180)) for 'precipitation'\n",
                         result.stdout)
