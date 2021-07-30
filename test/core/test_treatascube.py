from xcube.core.treatascube import merge_cube
from xcube.core.treatascube import split_cube
from xcube.core.treatascube import verify_cube_subset
from xcube.core.new import new_cube
from xcube.core.verify import assert_cube

import numpy as np
import xarray as xr
import unittest


class VerifyCubSubsetTest(unittest.TestCase):

    def test_all_well(self):
        cube = new_cube(variables=dict(x=1, y=2))
        try:
            verify_cube_subset(cube)
        except ValueError as ve:
            self.fail(f'No value error expected: {ve}')

    def test_no_vars(self):
        cube = new_cube(variables=None)
        with self.assertRaises(ValueError) as ve:
            verify_cube_subset(cube)
        self.assertEqual('Not at least one data variable '
                         'has spatial dimensions.',
                         f'{ve.exception}')

    def test_no_grid_mapping(self):
        cube = new_cube(variables=dict(x=1, y=2))
        cube = cube.drop_dims('lat')
        with self.assertRaises(ValueError) as ve:
            verify_cube_subset(cube)
        self.assertEqual('cannot find any grid mapping in dataset',
                         f'{ve.exception}')

    def test_no_time_info(self):
        cube = new_cube(drop_bounds=True, variables=dict(x=1, y=2))
        cube = cube.drop_vars('time')
        with self.assertRaises(ValueError) as ve:
            verify_cube_subset(cube)
        self.assertEqual('Dataset has no temporal information.',
                         f'{ve.exception}')


class SplitAndMergeTest(unittest.TestCase):

    def test_split(self):
        cube = new_cube(variables=dict(x=1, y=2))
        splitcube, removed_data_vars = split_cube(cube)
        self.assertEqual(dict(), removed_data_vars)
        self.assertEqual(cube.data_vars.keys(), splitcube.data_vars.keys())

    def test_split_remove_vars_and_merge(self):
        cube = new_cube(variables=dict(x=1, y=2))
        non_cube_dims = {}
        non_cube_dims['no_spatial_dims'] = \
            xr.DataArray([0.1, 0.2, 0.3, 0.4, 0.5],
                         dims=('time'))
        non_cube_dims['no_dims'] = np.array(b'', dtype='|S1')
        cube = cube.assign(non_cube_dims)

        with self.assertRaises(ValueError):
            assert_cube(cube)

        splitcube, removed_data_vars = split_cube(cube)
        self.assertEqual(non_cube_dims.keys(), removed_data_vars.keys())
        self.assertEqual(['x', 'y'], list(splitcube.data_vars.keys()))

        assert_cube(splitcube)

        merged_cube = merge_cube(splitcube, removed_data_vars)
        self.assertEqual(['x', 'y', 'no_spatial_dims', 'no_dims'],
                         list(merged_cube.data_vars.keys()))
