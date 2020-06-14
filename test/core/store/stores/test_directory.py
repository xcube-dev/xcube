import os.path
import unittest

from xcube.core.store import new_data_store
from xcube.core.store.stores.directory import DirectoryDataStore


class DirectoryCubeStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data_store = new_data_store('directory', base_dir='.')
        self.assertIsInstance(self.data_store, DirectoryDataStore)

    def test_get_data_ids(self):
        data_store = new_data_store('directory',
                                    base_dir=os.path.join(os.path.dirname(__file__),
                                                          '..', '..', '..', '..', 'examples', 'serve', 'demo'),
                                    read_only=True)
        self.assertEqual(
            {
                'cube-1-250-250.zarr',
                'cube-5-100-200.zarr',
                'cube-1-250-250.levels',
                'cube.nc'
            },
            set(data_store.get_data_ids()))
