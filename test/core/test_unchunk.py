import os.path
import unittest

from xcube.constants import FORMAT_NAME_ZARR
from xcube.core.chunk import chunk_dataset
from xcube.core.dsio import rimraf
from xcube.core.new import new_cube
from xcube.core.unchunk import unchunk_dataset


class UnchunkDatasetTest(unittest.TestCase):
    TEST_ZARR = 'test.zarr'

    def setUp(self):
        rimraf(self.TEST_ZARR)
        cube = new_cube(variables=dict(A=0.5, B=-1.5))
        cube = chunk_dataset(cube, chunk_sizes=dict(time=1, lat=90, lon=90), format_name=FORMAT_NAME_ZARR)
        cube.to_zarr(self.TEST_ZARR)

        self.chunked_a_files = {'.zarray',
                                '.zattrs',
                                '0.0.0', '0.0.1', '0.0.2', '0.0.3', '0.1.0', '0.1.1', '0.1.2', '0.1.3',
                                '1.0.0', '1.0.1', '1.0.2', '1.0.3', '1.1.0', '1.1.1', '1.1.2', '1.1.3',
                                '2.0.0', '2.0.1', '2.0.2', '2.0.3', '2.1.0', '2.1.1', '2.1.2', '2.1.3',
                                '3.0.0', '3.0.1', '3.0.2', '3.0.3', '3.1.0', '3.1.1', '3.1.2', '3.1.3',
                                '4.0.0', '4.0.1', '4.0.2', '4.0.3', '4.1.0', '4.1.1', '4.1.2', '4.1.3'}
        self.chunked_b_files = self.chunked_a_files
        self.chunked_time_files = {'.zarray', '.zattrs', '0', '1', '2', '3', '4'}
        self.chunked_lat_files = {'.zattrs', '.zarray', '0', '1'}
        self.chunked_lon_files = {'.zattrs', '.zarray', '0', '1', '2', '3'}

    def tearDown(self):
        rimraf(self.TEST_ZARR)

    def test_chunked(self):
        self._assert_cube_files(expected_a_files=self.chunked_a_files,
                                expected_b_files=self.chunked_b_files,
                                expected_time_files=self.chunked_time_files,
                                expected_lat_files=self.chunked_lat_files,
                                expected_lon_files=self.chunked_lon_files)

    def test_unchunk_all(self):
        unchunk_dataset(self.TEST_ZARR)
        self._assert_cube_files(expected_a_files={'.zarray', '.zattrs', '0.0.0'},
                                expected_b_files={'.zarray', '.zattrs', '0.0.0'},
                                expected_time_files={'.zarray', '.zattrs', '0'},
                                expected_lat_files={'.zarray', '.zattrs', '0'},
                                expected_lon_files={'.zarray', '.zattrs', '0'})

    def test_unchunk_coords_only(self):
        unchunk_dataset(self.TEST_ZARR, coords_only=True)
        self._assert_cube_files(expected_a_files=self.chunked_a_files,
                                expected_b_files=self.chunked_b_files,
                                expected_time_files={'.zarray', '.zattrs', '0'},
                                expected_lat_files={'.zarray', '.zattrs', '0'},
                                expected_lon_files={'.zarray', '.zattrs', '0'})

    def test_unchunk_data_var(self):
        unchunk_dataset(self.TEST_ZARR, var_names=['B'])
        self._assert_cube_files(expected_a_files=self.chunked_a_files,
                                expected_b_files={'.zarray', '.zattrs', '0.0.0'},
                                expected_time_files=self.chunked_time_files,
                                expected_lat_files=self.chunked_lat_files,
                                expected_lon_files=self.chunked_lon_files)

    def test_unchunk_coord_var(self):
        unchunk_dataset(self.TEST_ZARR, var_names=['time'], coords_only=True)
        self._assert_cube_files(expected_a_files=self.chunked_a_files,
                                expected_b_files=self.chunked_b_files,
                                expected_time_files={'.zarray', '.zattrs', '0'},
                                expected_lat_files=self.chunked_lat_files,
                                expected_lon_files=self.chunked_lon_files)

    def test_dont_unchunk_if_unchunked(self):
        unchunk_dataset(self.TEST_ZARR, var_names=['time'], coords_only=True)
        unchunk_dataset(self.TEST_ZARR, var_names=['time'], coords_only=True)
        unchunk_dataset(self.TEST_ZARR, var_names=['time'], coords_only=True)
        self._assert_cube_files(expected_a_files=self.chunked_a_files,
                                expected_b_files=self.chunked_b_files,
                                expected_time_files={'.zarray', '.zattrs', '0'},
                                expected_lat_files=self.chunked_lat_files,
                                expected_lon_files=self.chunked_lon_files)

    def test_unchunk_data_var_coords_only(self):
        with self.assertRaises(ValueError) as cm:
            unchunk_dataset(self.TEST_ZARR, var_names=['B'], coords_only=True)
        self.assertEqual("variable 'B' is not a coordinate variable in 'test.zarr'", f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            unchunk_dataset(self.TEST_ZARR, var_names=['C'], coords_only=False)
        self.assertEqual("variable 'C' is not a variable in 'test.zarr'", f'{cm.exception}')

    def test_unchunk_invalid_path(self):
        with self.assertRaises(ValueError) as cm:
            unchunk_dataset(self.TEST_ZARR + '.zip')
        self.assertEqual("'test.zarr.zip' is not a valid Zarr directory", f'{cm.exception}')

    def test_unchunk_invalid_vars(self):
        with self.assertRaises(ValueError) as cm:
            unchunk_dataset(self.TEST_ZARR, var_names=['times'], coords_only=True)
        self.assertEqual("variable 'times' is not a coordinate variable in 'test.zarr'", f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            unchunk_dataset(self.TEST_ZARR, var_names=['CHL'], coords_only=False)
        self.assertEqual("variable 'CHL' is not a variable in 'test.zarr'", f'{cm.exception}')

    def _assert_cube_files(self,
                           expected_a_files, expected_b_files,
                           expected_time_files, expected_lat_files, expected_lon_files):
        self._assert_files('A', expected_a_files)
        self._assert_files('B', expected_b_files)
        self._assert_files('time', expected_time_files)
        self._assert_files('lat', expected_lat_files)
        self._assert_files('lon', expected_lon_files)

    def _assert_files(self, var_name, expected_files):
        actual_files = os.listdir(os.path.join(self.TEST_ZARR, var_name))
        self.assertEqual(set(expected_files), set(actual_files))
