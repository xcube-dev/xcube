import unittest

import xarray as xr

from xcube.core.new import new_cube
from xcube.core.store.store import CubeStoreError
from xcube.core.store.stores.mem import MemoryCubeStore


class MemoryCubeStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.cube_store = MemoryCubeStore()
        self.cube_store.cubes.update({
            'cube_1': new_cube(variables=dict(B01=0.4, B02=0.5)),
            'cube_2': new_cube(variables=dict(B03=0.4, B04=0.5))
        })

    def test_iter_cubes(self):
        self.assertEqual({'cube_1', 'cube_2'},
                         set(cube_des.id for cube_des in self.cube_store.iter_cubes()))

    def test_open_cube(self):
        cube_1 = self.cube_store.open_cube('cube_1')
        self.assertIsInstance(cube_1, xr.Dataset)
        self.assertEqual({'B01', 'B02'}, set(map(str, cube_1.data_vars.keys())))
        cube_2 = self.cube_store.open_cube('cube_2')
        self.assertIsInstance(cube_2, xr.Dataset)
        self.assertEqual({'B03', 'B04'}, set(map(str, cube_2.data_vars.keys())))
        with self.assertRaises(CubeStoreError) as cm:
            self.cube_store.open_cube('cube_3')
        self.assertEqual('Unknown cube identifier "cube_3"', f'{cm.exception}')

    def test_write_and_delete_cube(self):
        cube_3 = new_cube(variables=dict(B05=0.1, B06=0.2))
        cube_3_id = self.cube_store.write_cube(cube_3, cube_id='cube_3')
        self.assertEqual('cube_3', cube_3_id)
        self.assertIs(cube_3, self.cube_store.open_cube(cube_3_id))

        cube_4 = new_cube(variables=dict(B07=0.1, B08=0.2))
        cube_4_id = self.cube_store.write_cube(cube_4)
        self.assertIsInstance(cube_4_id, str)
        self.assertIs(cube_4, self.cube_store.open_cube(cube_4_id))

        with self.assertRaises(CubeStoreError) as cm:
            self.cube_store.write_cube(cube_4, cube_id='cube_3')
        self.assertEqual('A cube named "cube_3" already exists', f'{cm.exception}')

        self.assertTrue(self.cube_store.delete_cube(cube_3_id))
        self.assertFalse(self.cube_store.delete_cube(cube_3_id))

        self.assertTrue(self.cube_store.delete_cube(cube_4_id))
        self.assertFalse(self.cube_store.delete_cube(cube_4_id))
