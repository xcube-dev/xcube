import numpy as np
import os.path
import sys

import xarray as xr

from test.cli.helpers import CliTest
from xcube.cli.prune import _delete_block_file
from xcube.core.dsio import rimraf
from xcube.core.dsio import write_cube
from xcube.core.new import new_cube
from xcube.core.verify import assert_cube


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
                                       temperature=np.nan)) \
            .chunk(dict(time=1, lat=90, lon=90))
        fv_encoding = dict(
            _FillValue=None
        )
        encoding = dict(
            precipitation=fv_encoding,
            temperature=fv_encoding
        )
        cube.to_zarr(self.TEST_CUBE, encoding=encoding)

    def tearDown(self) -> None:
        rimraf(self.TEST_CUBE)

    def test_dry_run(self):
        result = self.invoke_cli(['prune', self.TEST_CUBE, "-vv", "--dry-run"])
        self.assertEqual(0, result.exit_code)
        self.assertEqual(
            (
                "Opening dataset from 'test.zarr'...\n"
                "Identifying empty chunks...\n"
                "Found empty chunks in variable 'precipitation', "
                "deleting block files...\n"
                "Deleted 24 block file(s) for variable 'precipitation'.\n"
                "Found empty chunks in variable 'temperature', "
                "deleting block files...\n"
                "Deleted 24 block file(s) for variable 'temperature'.\n"
                "Done, 48 block file(s) deleted total.\n"
            ),
            result.stdout)
        expected_file_names = sorted([
            '.zarray',
            '.zattrs',
            '0.0.0', '0.0.1', '0.0.2', '0.0.3',
            '0.1.0', '0.1.1', '0.1.2', '0.1.3',
            '1.0.0', '1.0.1', '1.0.2', '1.0.3',
            '1.1.0', '1.1.1', '1.1.2', '1.1.3',
            '2.0.0', '2.0.1', '2.0.2', '2.0.3',
            '2.1.0', '2.1.1', '2.1.2', '2.1.3'
        ])
        self.assertEqual(expected_file_names,
                         sorted(os.listdir('test.zarr/precipitation')))
        self.assertEqual(expected_file_names,
                         sorted(os.listdir('test.zarr/temperature')))
        ds = xr.open_zarr('test.zarr')
        assert_cube(ds)
        self.assertIn('precipitation', ds)
        self.assertEqual((3, 180, 360), ds.precipitation.shape)
        self.assertEqual(('time', 'lat', 'lon'), ds.precipitation.dims)
        self.assertIn('temperature', ds)
        self.assertEqual((3, 180, 360), ds.temperature.shape)
        self.assertEqual(('time', 'lat', 'lon'), ds.temperature.dims)

    def test_no_dry_run(self):
        result = self.invoke_cli(['prune', self.TEST_CUBE, "-vv"])
        self.assertEqual(0, result.exit_code)
        self.assertEqual(
            (
                "Opening dataset from 'test.zarr'...\n"
                "Identifying empty chunks...\n"
                "Found empty chunks in variable 'precipitation', "
                "deleting block files...\n"
                "Deleted 24 block file(s) for variable 'precipitation'.\n"
                "Found empty chunks in variable 'temperature', "
                "deleting block files...\n"
                "Deleted 24 block file(s) for variable 'temperature'.\n"
                "Done, 48 block file(s) deleted total.\n"
            ),
            result.stdout)
        expected_file_names = sorted(['.zarray', '.zattrs'])
        self.assertEqual(expected_file_names,
                         sorted(os.listdir('test.zarr/precipitation')))
        self.assertEqual(expected_file_names,
                         sorted(os.listdir('test.zarr/temperature')))
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

        def monitor(message, level):
            nonlocal actual_message
            actual_message = message

        actual_message = None
        ok = _delete_block_file(self.TEST_CUBE,
                                'precipitation', (0, 3, 76),
                                True, monitor=monitor)
        self.assertFalse(ok)
        file_path = os.path.join(self.TEST_CUBE,
                                 'precipitation', '0.3.76')
        dir_path = os.path.join(self.TEST_CUBE,
                                'precipitation', '0', '3', '76')
        self.assertEqual(f"Error: could find neither block file "
                         f"{file_path} nor "
                         f"{dir_path}", actual_message)

        block_file = os.path.join(self.TEST_CUBE,
                                  'precipitation', '1.1.0')

        actual_message = None
        ok = _delete_block_file(self.TEST_CUBE,
                                'precipitation', (1, 1, 0),
                                True, monitor=monitor)
        self.assertTrue(ok)
        self.assertEqual(f'Deleting chunk file {block_file}',
                         actual_message)

        actual_message = None
        ok = _delete_block_file(self.TEST_CUBE,
                                'precipitation', (1, 1, 0),
                                False, monitor=monitor)
        self.assertTrue(ok)
        self.assertEqual(f'Deleting chunk file {block_file}',
                         actual_message)
        self.assertFalse(os.path.exists(block_file))

        if sys.platform == 'win32':
            block_file = os.path.join(self.TEST_CUBE,
                                      'precipitation', '1.1.1')
            # Open block, so we cannot delete (Windows only)
            # noinspection PyUnusedLocal
            with open(block_file, 'wb') as fp:
                actual_message = None
                ok = _delete_block_file(self.TEST_CUBE,
                                        'precipitation', (1, 1, 1),
                                        False, monitor=monitor)
                self.assertFalse(ok)
                self.assertIsNotNone(actual_message)
                # noinspection PyUnresolvedReferences
                self.assertTrue(actual_message.startswith(
                    f'Error: failed to delete block file {block_file}: '
                ))
