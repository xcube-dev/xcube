import unittest

import xarray as xr

from test.s3test import MOTO_SERVER_ENDPOINT_URL
from test.s3test import S3Test
from xcube.core.new import new_cube
from xcube.core.store import MutableDataStore
from xcube.core.store.fs.registry import new_fs_data_store
from xcube.util.temp import new_temp_dir


class FsDataStoresTest(unittest.TestCase):
    # TODO: add gpd.GeoDataFrame tests for "file", "memory", "s3"

    def assertDataStoreWriteOpenDeleteDataset(self, data_store, ext):
        self.assertIsInstance(data_store, MutableDataStore)

        self.assertEqual([], list(data_store.get_data_ids()))

        data_id = f'ds{ext}'

        data = new_cube(variables=dict(A=8, B=9))
        data_store.write_data(data, data_id)
        self.assertEqual([data_id], list(data_store.get_data_ids()))

        data = data_store.open_data(data_id)
        self.assertIsInstance(data, xr.Dataset)

        try:
            data_store.delete_data(data_id)
        except PermissionError:  # Typically occurs on win32 due to fsspec
            return
        self.assertEqual([], list(data_store.get_data_ids()))


class FileFsDataStoresTest(FsDataStoresTest):

    def test_dataset_zarr(self):
        data_store = new_fs_data_store('file',
                                       root=new_temp_dir(prefix='xcube-test'))
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.zarr')

    def test_dataset_netcdf(self):
        data_store = new_fs_data_store('file',
                                       root=new_temp_dir(prefix='xcube-test'))
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.nc')


class MemoryFsDataStoresTest(FsDataStoresTest):

    def test_dataset_zarr(self):
        data_store = new_fs_data_store('memory',
                                       root='xcube-test')
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.zarr')

    def test_dataset_netcdf(self):
        data_store = new_fs_data_store('memory',
                                       root='xcube-test')
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.nc')


class S3FsDataStoresTest(S3Test, FsDataStoresTest):
    fs_params = dict(
        anon=False,
        client_kwargs=dict(
            endpoint_url=MOTO_SERVER_ENDPOINT_URL,
        )
    )

    def test_dataset_zarr(self):
        data_store = new_fs_data_store('s3',
                                       root='xcube-test',
                                       fs_params=self.fs_params)
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.zarr')

    def test_dataset_netcdf(self):
        data_store = new_fs_data_store('s3',
                                       root='xcube-test',
                                       fs_params=self.fs_params)
        self.assertDataStoreWriteOpenDeleteDataset(data_store, '.nc')
