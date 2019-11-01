import os
import os.path
import unittest
from typing import Set

from xcube.core import new_cube
from xcube.util.chunk import chunk_dataset
from xcube.util.constants import FORMAT_NAME_ZARR
from xcube.util.dsio import rimraf
from xcube.util.optimize import optimize_dataset

TEST_CUBE = chunk_dataset(new_cube(time_periods=3, variables=dict(A=0.5, B=-1.5)),
                          chunk_sizes=dict(time=1, lat=180, lon=360), format_name=FORMAT_NAME_ZARR)

TEST_CUBE_ZARR = 'test.zarr'

TEST_CUBE_FILE_SET = {
    '.zattrs', '.zgroup',
    'A/.zarray', 'A/.zattrs', 'A/0.0.0', 'A/1.0.0', 'A/2.0.0',
    'B/.zarray', 'B/.zattrs', 'B/0.0.0', 'B/1.0.0', 'B/2.0.0',
    'lat/.zarray', 'lat/.zattrs', 'lat/0',
    'lat_bnds/.zarray', 'lat_bnds/.zattrs', 'lat_bnds/0.0',
    'lon/.zarray', 'lon/.zattrs', 'lon/0',
    'lon_bnds/.zarray', 'lon_bnds/.zattrs', 'lon_bnds/0.0',
    'time/.zarray', 'time/.zattrs', 'time/0', 'time/1', 'time/2',
    'time_bnds/.zarray', 'time_bnds/.zattrs', 'time_bnds/0.0', 'time_bnds/1.0', 'time_bnds/2.0'
}


class OptimizeDatasetTest(unittest.TestCase):

    def setUp(self):
        rimraf(TEST_CUBE_ZARR)
        TEST_CUBE.to_zarr(TEST_CUBE_ZARR)

    def tearDown(self):
        rimraf(TEST_CUBE_ZARR)

    def test_optimize_dataset_in_place(self):
        self.assertEqual(TEST_CUBE_FILE_SET, list_file_set(TEST_CUBE_ZARR))

        optimize_dataset(TEST_CUBE_ZARR, in_place=True)

        expected_files = set(TEST_CUBE_FILE_SET)
        expected_files.add('.zmetadata')
        self.assertEqual(expected_files, list_file_set(TEST_CUBE_ZARR))

    def test_optimize_dataset_in_place_unchunk_coords(self):
        self.assertEqual(TEST_CUBE_FILE_SET, list_file_set(TEST_CUBE_ZARR))

        optimize_dataset(TEST_CUBE_ZARR, in_place=True, unchunk_coords=True)

        expected_files = set(TEST_CUBE_FILE_SET)
        expected_files.add('.zmetadata')
        expected_files.remove('time/1')
        expected_files.remove('time/2')
        expected_files.remove('time_bnds/1.0')
        expected_files.remove('time_bnds/2.0')
        self.assertEqual(expected_files, list_file_set(TEST_CUBE_ZARR))

    def test_failures(self):
        with self.assertRaises(RuntimeError) as cm:
            optimize_dataset('pippo', in_place=True, exception_type=RuntimeError)
        self.assertEqual('Input path must point to ZARR dataset directory.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            optimize_dataset(TEST_CUBE_ZARR, exception_type=RuntimeError)
        self.assertEqual('Output path must be given.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            optimize_dataset(TEST_CUBE_ZARR, output_path=TEST_CUBE_ZARR, exception_type=RuntimeError)
        self.assertEqual('Output path already exists.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            optimize_dataset(TEST_CUBE_ZARR, output_path='./' + TEST_CUBE_ZARR, exception_type=RuntimeError)
        self.assertEqual('Output path already exists.', f'{cm.exception}')


def list_file_set(dir_path: str) -> Set[str]:
    """ Get set of all files in a directory and all of its sub-directories. """
    actual_files = set()
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            rel_root = root[len(dir_path) + 1:]
            if rel_root:
                actual_files.add(rel_root + '/' + f)
            else:
                actual_files.add(f)
    return actual_files
