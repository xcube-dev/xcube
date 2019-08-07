import os
import os.path
import unittest

from xcube.api import new_cube
from xcube.util.chunk import chunk_dataset
from xcube.util.consolidate import consolidate_dataset
from xcube.util.constants import FORMAT_NAME_ZARR
from xcube.util.dsio import rimraf


class ConsolidateDatasetTest(unittest.TestCase):
    TEST_ZARR = 'test.zarr'

    def setUp(self):
        rimraf(self.TEST_ZARR)
        cube = new_cube(time_periods=3, variables=dict(A=0.5, B=-1.5))
        cube = chunk_dataset(cube, chunk_sizes=dict(time=1, lat=180, lon=360), format_name=FORMAT_NAME_ZARR)
        cube.to_zarr(self.TEST_ZARR)

        self.expected_files = [
            '.zattrs', '.zgroup',
            'A/.zarray', 'A/.zattrs', 'A/0.0.0', 'A/1.0.0', 'A/2.0.0',
            'B/.zarray', 'B/.zattrs', 'B/0.0.0', 'B/1.0.0', 'B/2.0.0',
            'lat/.zarray', 'lat/.zattrs', 'lat/0',
            'lat_bnds/.zarray', 'lat_bnds/.zattrs', 'lat_bnds/0.0',
            'lon/.zarray', 'lon/.zattrs', 'lon/0',
            'lon_bnds/.zarray', 'lon_bnds/.zattrs', 'lon_bnds/0.0',
            'time/.zarray', 'time/.zattrs', 'time/0', 'time/1', 'time/2',
            'time_bnds/.zarray', 'time_bnds/.zattrs', 'time_bnds/0.0', 'time_bnds/1.0', 'time_bnds/2.0'
        ]

    def tearDown(self):
        rimraf(self.TEST_ZARR)

    def test_consolidate_dataset_wo_coords(self):
        actual_files = self._list_files(self.TEST_ZARR)
        self.assertEqual(set(self.expected_files), set(actual_files))

        consolidate_dataset(self.TEST_ZARR, in_place=True)

        actual_files = self._list_files(self.TEST_ZARR)
        expected_files = ['.zmetadata'] + self.expected_files
        self.assertEqual(set(expected_files), set(actual_files))

    def test_consolidate_dataset_with_coords(self):
        actual_files = self._list_files(self.TEST_ZARR)
        self.assertEqual(set(self.expected_files), set(actual_files))

        consolidate_dataset(self.TEST_ZARR, in_place=True, unchunk_coords=True)

        actual_files = self._list_files(self.TEST_ZARR)
        expected_files = [
            '.zattrs', '.zgroup', '.zmetadata',
            'A/.zarray', 'A/.zattrs', 'A/0.0.0', 'A/1.0.0', 'A/2.0.0',
            'B/.zarray', 'B/.zattrs', 'B/0.0.0', 'B/1.0.0', 'B/2.0.0',
            'lat/.zarray', 'lat/.zattrs', 'lat/0',
            'lat_bnds/.zarray', 'lat_bnds/.zattrs', 'lat_bnds/0.0',
            'lon/.zarray', 'lon/.zattrs', 'lon/0',
            'lon_bnds/.zarray', 'lon_bnds/.zattrs', 'lon_bnds/0.0',
            'time/.zarray', 'time/.zattrs', 'time/0',
            'time_bnds/.zarray', 'time_bnds/.zattrs', 'time_bnds/0.0'
        ]
        self.assertEqual(set(expected_files), set(actual_files))

    @classmethod
    def _list_files(cls, path):
        actual_files = list()
        for root, dirs, files in os.walk(path):
            for f in files:
                rel_root = root[len(path) + 1:]
                if rel_root:
                    actual_files.append(rel_root + '/' + f)
                else:
                    actual_files.append(f)
        return actual_files
