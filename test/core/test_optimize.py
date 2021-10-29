import os
import os.path
import unittest
from typing import Set, Union, Sequence

from xcube.constants import FORMAT_NAME_ZARR
from xcube.core.chunk import chunk_dataset
from xcube.core.dsio import rimraf
from xcube.core.new import new_cube
from xcube.core.optimize import optimize_dataset

TEST_CUBE = chunk_dataset(new_cube(time_periods=3, variables=dict(A=0.5, B=-1.5)),
                          chunk_sizes=dict(time=1, lat=180, lon=360), format_name=FORMAT_NAME_ZARR)

INPUT_CUBE_PATH = 'test.zarr'
OUTPUT_CUBE_PATH = 'test_opt.zarr'
OUTPUT_CUBE_PATTERN = '{input}_opt.zarr'

INPUT_CUBE_FILE_SET = {
    '.zattrs', '.zgroup',  '.zmetadata',
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
    def _clear_outputs(self):
        rimraf(INPUT_CUBE_PATH)
        rimraf(OUTPUT_CUBE_PATH)

    def setUp(self):
        self._clear_outputs()
        TEST_CUBE.to_zarr(INPUT_CUBE_PATH)
        self.assertEqual(INPUT_CUBE_FILE_SET, list_file_set(INPUT_CUBE_PATH))

    def tearDown(self):
        self._clear_outputs()

    def test_optimize_dataset(self):
        self._test_optimize_dataset(unchunk_coords=False, in_place=False,
                                    expected_output_path=OUTPUT_CUBE_PATH,
                                    expected_cons_time=False,
                                    expected_cons_time_bnds=False)

    def test_optimize_dataset_in_place(self):
        self._test_optimize_dataset(unchunk_coords=False, in_place=True,
                                    expected_output_path=INPUT_CUBE_PATH,
                                    expected_cons_time=False,
                                    expected_cons_time_bnds=False)

    def test_optimize_dataset_unchunk_coords_true(self):
        self._test_optimize_dataset(unchunk_coords=True, in_place=False,
                                    expected_output_path=OUTPUT_CUBE_PATH,
                                    expected_cons_time=True,
                                    expected_cons_time_bnds=True)

    def test_optimize_dataset_unchunk_coords_true_in_place(self):
        self._test_optimize_dataset(unchunk_coords=True, in_place=True,
                                    expected_output_path=INPUT_CUBE_PATH,
                                    expected_cons_time=True,
                                    expected_cons_time_bnds=True)

    def test_optimize_dataset_unchunk_coords_str(self):
        self._test_optimize_dataset(unchunk_coords='time', in_place=False,
                                    expected_output_path=OUTPUT_CUBE_PATH,
                                    expected_cons_time=True,
                                    expected_cons_time_bnds=False)

    def test_optimize_dataset_unchunk_coords_str_in_place(self):
        self._test_optimize_dataset(unchunk_coords='time', in_place=True,
                                    expected_output_path=INPUT_CUBE_PATH,
                                    expected_cons_time=True,
                                    expected_cons_time_bnds=False)

    def test_optimize_dataset_unchunk_coords_str_list(self):
        self._test_optimize_dataset(unchunk_coords=['time_bnds'], in_place=False,
                                    expected_output_path=OUTPUT_CUBE_PATH,
                                    expected_cons_time=False,
                                    expected_cons_time_bnds=True)

    def test_optimize_dataset_unchunk_coords_str_list_in_place(self):
        self._test_optimize_dataset(unchunk_coords=['time_bnds'], in_place=True,
                                    expected_output_path=INPUT_CUBE_PATH,
                                    expected_cons_time=False,
                                    expected_cons_time_bnds=True)

    def test_optimize_dataset_unchunk_coords_str_tuple(self):
        self._test_optimize_dataset(unchunk_coords=('time_bnds', 'time'), in_place=False,
                                    expected_output_path=OUTPUT_CUBE_PATH,
                                    expected_cons_time=True,
                                    expected_cons_time_bnds=True)

    def test_optimize_dataset_unchunk_coords_str_tuple_in_place(self):
        self._test_optimize_dataset(unchunk_coords=('time_bnds', 'time'), in_place=True,
                                    expected_output_path=INPUT_CUBE_PATH,
                                    expected_cons_time=True,
                                    expected_cons_time_bnds=True)

    def _test_optimize_dataset(self, unchunk_coords: Union[bool, str, Sequence[str]], in_place: bool,
                               expected_output_path: str, expected_cons_time: bool = False,
                               expected_cons_time_bnds: bool = False):
        if not in_place:
            optimize_dataset(INPUT_CUBE_PATH, output_path=OUTPUT_CUBE_PATTERN,
                             in_place=in_place, unchunk_coords=unchunk_coords)
        else:
            optimize_dataset(INPUT_CUBE_PATH,
                             in_place=in_place, unchunk_coords=unchunk_coords)
        self._assert_consolidated(expected_output_path, expected_cons_time, expected_cons_time_bnds)

    def _assert_consolidated(self,
                             cube_path: str,
                             cons_time: bool = False,
                             cons_time_bnds: bool = False):
        self.assertTrue(os.path.isdir(cube_path))
        expected_files = set(INPUT_CUBE_FILE_SET)
        expected_files.add('.zmetadata')
        if cons_time:
            expected_files.remove('time/1')
            expected_files.remove('time/2')
        if cons_time_bnds:
            expected_files.remove('time_bnds/1.0')
            expected_files.remove('time_bnds/2.0')
        self.assertEqual(expected_files, list_file_set(cube_path))

    def test_failures(self):
        with self.assertRaises(RuntimeError) as cm:
            optimize_dataset('pippo', in_place=True, exception_type=RuntimeError)
        self.assertEqual('Input path must point to ZARR dataset directory.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            optimize_dataset(INPUT_CUBE_PATH, exception_type=RuntimeError)
        self.assertEqual('Output path must be given.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            optimize_dataset(INPUT_CUBE_PATH, output_path=INPUT_CUBE_PATH, exception_type=RuntimeError)
        self.assertEqual('Output path already exists.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            optimize_dataset(INPUT_CUBE_PATH, output_path='./' + INPUT_CUBE_PATH, exception_type=RuntimeError)
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
