import unittest

import xarray as xr

from xcube.core.store import MutableDataStore
from xcube.core.store.fs.registry import new_fs_data_store
from xcube.util.temp import new_temp_dir


class FsDataStoresInRegistryTest(unittest.TestCase):

    def test_dataset_zarr_file(self):
        data_store = new_fs_data_store('file', root=new_temp_dir())
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.zarr')

    def test_dataset_zarr_memory(self):
        data_store = new_fs_data_store('memory')
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.zarr')

    def test_dataset_netcdf_file(self):
        data_store = new_fs_data_store('file', root=new_temp_dir())
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.nc')

    def test_dataset_netcdf_memory(self):
        data_store = new_fs_data_store('memory')
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.nc')

    # TODO: add xr.Dataset tests for "s3"
    # TODO: add gpd.GeoDataFrame tests for "file", "memory", "s3"

    def assertDataStoreWriteOpenDeleteDataset(self, data_store, ext):
        self.assertIsInstance(data_store, MutableDataStore)

        self.assertEqual([], list(data_store.get_data_ids()))

        data_id = f'ds{ext}'

        data_store.write_data(xr.Dataset(), data_id)
        self.assertEqual([data_id], list(data_store.get_data_ids()))

        data = data_store.open_data(data_id)
        self.assertIsInstance(data, xr.Dataset)

        data_store.delete_data(data_id)
        self.assertEqual([], list(data_store.get_data_ids()))
