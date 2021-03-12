import unittest
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.store import DataStoreError
from xcube.core.store.stores.memory import MemoryDataStore


class MemoryCubeStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data_store = MemoryDataStore()
        self.data_store.data_dict.update({
            'cube_1': new_cube(variables=dict(B01=0.4, B02=0.5)),
            'cube_2': new_cube(variables=dict(B03=0.4, B04=0.5))
        })

    def test_get_data_ids(self):
        self.assertEqual({'cube_1', 'cube_2'},
                         set(self.data_store.get_data_ids()))
        list_with_attrs = list(self.data_store.get_data_ids(include_attrs=[]))
        self.assertEqual(3, len(list_with_attrs))
        self.assertIn(('cube_1', {}), list_with_attrs)
        self.assertIn(('cube_2', {}), list_with_attrs)

    def test_open_data(self):
        cube_1 = self.data_store.open_data('cube_1')
        self.assertIsInstance(cube_1, xr.Dataset)
        self.assertEqual({'B01', 'B02'}, set(map(str, cube_1.data_vars.keys())))
        cube_2 = self.data_store.open_data('cube_2')
        self.assertIsInstance(cube_2, xr.Dataset)
        self.assertEqual({'B03', 'B04'}, set(map(str, cube_2.data_vars.keys())))
        with self.assertRaises(DataStoreError) as cm:
            self.data_store.open_data('cube_3')
        self.assertEqual('Data resource "cube_3" does not exist in store', f'{cm.exception}')

    def test_write_and_delete_data(self):
        cube_3 = new_cube(variables=dict(B05=0.1, B06=0.2))
        cube_3_id = self.data_store.write_data(cube_3, data_id='cube_3')
        self.assertEqual('cube_3', cube_3_id)
        self.assertIs(cube_3, self.data_store.open_data(cube_3_id))

        cube_4 = new_cube(variables=dict(B07=0.1, B08=0.2))
        cube_4_id = self.data_store.write_data(cube_4)
        self.assertIsInstance(cube_4_id, str)
        self.assertIs(cube_4, self.data_store.open_data(cube_4_id))

        with self.assertRaises(DataStoreError) as cm:
            self.data_store.write_data(cube_4, data_id='cube_3')
        self.assertEqual('Data resource "cube_3" already exist in store', f'{cm.exception}')

        self.data_store.delete_data(cube_3_id)
        self.data_store.delete_data(cube_4_id)

        with self.assertRaises(DataStoreError) as cm:
            self.data_store.delete_data(cube_3_id)
        self.assertEqual('Data resource "cube_3" does not exist in store', f'{cm.exception}')

        with self.assertRaises(DataStoreError) as cm:
            self.data_store.delete_data(cube_4_id)
        self.assertEqual(f'Data resource "{cube_4_id}" does not exist in store', f'{cm.exception}')
