import os.path

import numpy as np
import xarray as xr

from test.cli.helpers import CliTest
from xcube.api import assert_cube, new_cube
from xcube.api.readwrite import write_cube
from xcube.cli.prune import _delete_block_file
from xcube.util.dsio import rimraf


class PruneTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['prune', '--help'])
        self.assertEqual(0, result.exit_code)


class PruneDataTest(CliTest):
    TEST_CUBE = "test.zarr"

    def setUp(self) -> None:
        rimraf(self.TEST_CUBE)
        cube = new_cube(time_periods=3,
                        variables=dict(precipitation=np.nan,
                                       temperature=np.nan)).chunk(dict(time=1, lat=90, lon=90))

        write_cube(cube, self.TEST_CUBE, "zarr", cube_asserted=True)

    def tearDown(self) -> None:
        rimraf(self.TEST_CUBE)

    def test_dry_run(self):
        result = self.invoke_cli(['prune', self.TEST_CUBE, "--dry-run"])
        self.assertEqual(0, result.exit_code)
        self.assertEqual("Opening cube from 'test.zarr'...\n"
                         "Identifying empty blocks...\n"
                         "Deleting 24 empty block file(s) for variable 'precipitation'...\n"
                         "Deleting 24 empty block file(s) for variable 'temperature'...\n"
                         "Done, 48 block file(s) deleted.\n",
                         result.stdout)
        expected_file_names = ['.zarray',
                               '.zattrs',
                               '0.0.0', '0.0.1', '0.0.2', '0.0.3', '0.1.0', '0.1.1', '0.1.2', '0.1.3',
                               '1.0.0', '1.0.1', '1.0.2', '1.0.3', '1.1.0', '1.1.1', '1.1.2', '1.1.3',
                               '2.0.0', '2.0.1', '2.0.2', '2.0.3', '2.1.0', '2.1.1', '2.1.2', '2.1.3']
        self.assertEqual(expected_file_names, os.listdir('test.zarr/precipitation'))
        self.assertEqual(expected_file_names, os.listdir('test.zarr/temperature'))
        ds = xr.open_zarr('test.zarr')
        assert_cube(ds)
        self.assertIn('precipitation', ds)
        self.assertEqual((3, 180, 360), ds.precipitation.shape)
        self.assertEqual(('time', 'lat', 'lon'), ds.precipitation.dims)
        self.assertIn('temperature', ds)
        self.assertEqual((3, 180, 360), ds.temperature.shape)
        self.assertEqual(('time', 'lat', 'lon'), ds.temperature.dims)

    def test_no_dry_run(self):
        result = self.invoke_cli(['prune', self.TEST_CUBE])
        self.assertEqual(0, result.exit_code)
        self.assertEqual("Opening cube from 'test.zarr'...\n"
                         "Identifying empty blocks...\n"
                         "Deleting 24 empty block file(s) for variable 'precipitation'...\n"
                         "Deleting 24 empty block file(s) for variable 'temperature'...\n"
                         "Done, 48 block file(s) deleted.\n",
                         result.stdout)
        expected_file_names = ['.zarray', '.zattrs']
        self.assertEqual(expected_file_names, os.listdir('test.zarr/precipitation'))
        self.assertEqual(expected_file_names, os.listdir('test.zarr/temperature'))
        ds = xr.open_zarr('test.zarr')
        assert_cube(ds)
        self.assertIn('precipitation', ds)
        self.assertEqual((3, 180, 360), ds.precipitation.shape)
        self.assertEqual(('time', 'lat', 'lon'), ds.precipitation.dims)
        self.assertIn('temperature', ds)
        self.assertEqual((3, 180, 360), ds.temperature.shape)
        self.assertEqual(('time', 'lat', 'lon'), ds.temperature.dims)

    def test_delete_block_file(self):
        actual_message = None

        def monitor(message):
            nonlocal actual_message
            actual_message = message

        actual_message = None
        ok = _delete_block_file(self.TEST_CUBE, 'precipitation', (0, 3, 76), True, monitor=monitor)
        self.assertFalse(ok)
        self.assertEqual(f"error: could neither find block file "
                         f"{os.path.join(self.TEST_CUBE, 'precipitation', '0.3.76')} nor "
                         f"{os.path.join(self.TEST_CUBE, 'precipitation', '0', '3', '76')}", actual_message)

        actual_message = None
        ok = _delete_block_file(self.TEST_CUBE, 'precipitation', (1, 1, 0), True, monitor=monitor)
        self.assertTrue(ok)
        self.assertEqual(None, actual_message)

        actual_message = None
        ok = _delete_block_file(self.TEST_CUBE, 'precipitation', (1, 1, 0), False, monitor=monitor)
        self.assertTrue(ok)
        self.assertEqual(None, actual_message)
        self.assertFalse(os.path.exists(os.path.join(self.TEST_CUBE, 'precipitation', '1.1.0')))

        # Open block, so we cannot delete
        path = os.path.join(self.TEST_CUBE, 'precipitation', '1.1.1')
        with open(path, 'wb'):
            actual_message = None
            ok = _delete_block_file(self.TEST_CUBE, 'precipitation', (1, 1, 1), False, monitor=monitor)
            self.assertFalse(ok)
            self.assertIsNotNone(actual_message)
            self.assertTrue(actual_message.startswith(f'error: failed to delete block file {path}: '))
