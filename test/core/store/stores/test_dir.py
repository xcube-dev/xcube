import os.path
import unittest

from xcube.core.store.store import new_data_store
from xcube.core.store.stores.dir import DirectoryDataStore


class DirectoryCubeStoreTest(unittest.TestCase):
    def test_new_data_store(self):
        store = new_data_store('dir', base_dir='.')
        self.assertIsInstance(store, DirectoryDataStore)

    def test_get_data_ids(self):
        data_store = DirectoryDataStore(
            base_dir=os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'examples', 'serve', 'demo'),
            read_only=True)
        self.assertEqual({'cube-1-250-250.zarr', 'cube-5-100-200.zarr', 'cube-1-250-250.levels', 'cube.nc'},
                         set(data_store.get_data_ids()))
