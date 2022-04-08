import os.path
from typing import List

import xarray as xr

from test.cli.helpers import CliTest, CliDataTest, TEST_ZARR_DIR
from xcube.core.verify import assert_cube


class ResampleTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['resample', '--help'])
        self.assertEqual(0, result.exit_code)


class ResampleDataTest(CliDataTest):
    def outputs(self) -> List[str]:
        return ['out.zarr', 'resampled.zarr']

    def test_all_defaults(self):
        result = self.invoke_cli(['resample', '-v', TEST_ZARR_DIR])
        self.assertEqual(0, result.exit_code)
        self.assertEqual("Opening cube from 'test.zarr'...\n"
                         "Resampling...\n"
                         "Writing resampled cube to 'out.zarr'...\n"
                         "Done.\n",
                         result.stderr)
        self.assertTrue(os.path.isdir('out.zarr'))
        ds = xr.open_zarr('out.zarr')
        assert_cube(ds)
        self.assertIn('precipitation_mean', ds)
        self.assertIn('temperature_mean', ds)
        self.assertIn('soil_moisture_mean', ds)

    def test_with_output(self):
        result = self.invoke_cli(['resample', TEST_ZARR_DIR, '--output', 'resampled.zarr'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir('resampled.zarr'))
        ds = xr.open_zarr('resampled.zarr')
        assert_cube(ds)
        self.assertIn('precipitation_mean', ds)
        self.assertIn('temperature_mean', ds)
        self.assertIn('soil_moisture_mean', ds)

    def test_with_vars(self):
        result = self.invoke_cli(['resample', TEST_ZARR_DIR, '--vars', 'temperature,precipitation'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir('out.zarr'))
        ds = xr.open_zarr('out.zarr')
        assert_cube(ds)
        self.assertIn('precipitation_mean', ds)
        self.assertIn('temperature_mean', ds)
        self.assertNotIn('soil_moisture_mean', ds)

    def test_downsample_with_multiple_methods(self):
        result = self.invoke_cli(['resample',
                                  '--variables', 'temperature',
                                  '-F', 'all',
                                  '-M', 'mean',
                                  '-M', 'count',
                                  '-M', 'prod',
                                  TEST_ZARR_DIR])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir('out.zarr'))
        ds = xr.open_zarr('out.zarr')
        assert_cube(ds)
        self.assertIn('temperature_mean', ds)
        self.assertIn('temperature_count', ds)
        self.assertIn('temperature_prod', ds)

    def test_upsample_with_multiple_methods(self):
        result = self.invoke_cli(['resample',
                                  '--variables', 'temperature',
                                  '-F', '12H',
                                  '-T', '6H',
                                  # '-K', 'quadratic',
                                  # '-M', 'interpolate',
                                  '-M', 'nearest',
                                  TEST_ZARR_DIR])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir('out.zarr'))
        ds = xr.open_zarr('out.zarr')
        assert_cube(ds)
        # self.assertIn('temperature_interpolate', ds)
        self.assertIn('temperature_nearest', ds)
