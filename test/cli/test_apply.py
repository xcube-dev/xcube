import os
from typing import List

import xarray as xr

from test.cli.test_cli import CliDataTest, TEST_ZARR_DIR

INPUT_PATH = TEST_ZARR_DIR
OUTPUT_PATH = 'output.zarr'


class ApplyCliTest(CliDataTest):

    def outputs(self) -> List[str]:
        return [OUTPUT_PATH]

    def test_help_option(self):
        result = self.invoke_cli(['apply', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_apply_with_init(self):
        self._test_script(os.path.join(os.path.dirname(__file__), 'apply-scripts', 'with-init.py'))

    def test_apply_without_init(self):
        self._test_script(os.path.join(os.path.dirname(__file__), 'apply-scripts', 'without-init.py'))

    def _test_script(self, script_path):
        result = self.invoke_cli(['apply',
                                  '--dask', 'parallelized',
                                  '--params', 'a=0.1,b=0.4',
                                  '--vars', 'precipitation,soil_moisture',
                                  '--output', OUTPUT_PATH,
                                  script_path,
                                  INPUT_PATH])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(OUTPUT_PATH))

        ds = xr.open_zarr(OUTPUT_PATH)
        self.assertEqual(['output'], list(ds.data_vars))
        self.assertAlmostEqual(0.45, float(ds.output.mean()))
