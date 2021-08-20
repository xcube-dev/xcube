import unittest
from typing import Type, Union

import xarray as xr

from test.s3test import MOTO_SERVER_ENDPOINT_URL
from test.s3test import S3Test
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.new import new_cube
from xcube.core.store import MutableDataStore, DatasetDescriptor, DataStoreError
from xcube.core.store.fs.registry import new_fs_data_store
from xcube.util.temp import new_temp_dir


class FsDataStoresTest(unittest.TestCase):
    # TODO: add assertGeoDataFrameSupport

    def assertDatasetSupport(self,
                             data_store: MutableDataStore,
                             filename_ext: str,
                             expected_type_specifier: str,
                             expected_type: Union[Type[xr.Dataset],
                                                  Type[MultiLevelDataset]]):
        """
        Call all DataStore operations to ensure data of type
        xr.Dataset//MultiLevelDataset is supported by *data_store*.

        :param data_store: The filesystem data store instance.
        :param filename_ext: Filename extension that identifies
            a supported dataset format.
        :param expected_type: The expected data type.
        """

        data_id = f'ds{filename_ext}'

        self.assertIsInstance(data_store, MutableDataStore)

        self.assertEqual({'dataset', 'dataset[multilevel]', 'geodataframe'},
                         set(data_store.get_type_specifiers()))

        with self.assertRaises(DataStoreError):
            data_store.get_type_specifiers_for_data(data_id)
        self.assertEqual(False, data_store.has_data(data_id))
        self.assertEqual([], list(data_store.get_data_ids()))

        data = new_cube(variables=dict(A=8, B=9))
        data_store.write_data(data, data_id)
        self.assertEqual({expected_type_specifier},
                         set(data_store.get_type_specifiers_for_data(data_id)))
        self.assertEqual(True, data_store.has_data(data_id))
        self.assertEqual([data_id], list(data_store.get_data_ids()))

        data_descriptors = list(data_store.search_data())
        self.assertEqual(1, len(data_descriptors))
        self.assertIsInstance(data_descriptors[0], DatasetDescriptor)

        data = data_store.open_data(data_id)
        self.assertIsInstance(data, expected_type)

        try:
            data_store.delete_data(data_id)
        except PermissionError:  # Typically occurs on win32 due to fsspec
            return
        with self.assertRaises(DataStoreError):
            data_store.get_type_specifiers_for_data(data_id)
        self.assertEqual(False, data_store.has_data(data_id))
        self.assertEqual([], list(data_store.get_data_ids()))


class FileFsDataStoresTest(FsDataStoresTest):

    def test_mldataset_levels(self):
        data_store = new_fs_data_store('file',
                                       root=new_temp_dir(prefix='xcube-test'))
        self.assertDatasetSupport(data_store, '.levels',
                                  'dataset[multilevel]',
                                  MultiLevelDataset)

    def test_dataset_zarr(self):
        data_store = new_fs_data_store('file',
                                       root=new_temp_dir(prefix='xcube-test'))
        self.assertDatasetSupport(data_store, '.zarr',
                                  'dataset',
                                  xr.Dataset)

    def test_dataset_netcdf(self):
        data_store = new_fs_data_store('file',
                                       root=new_temp_dir(prefix='xcube-test'))
        self.assertDatasetSupport(data_store, '.nc',
                                  'dataset',
                                  xr.Dataset)


class MemoryFsDataStoresTest(FsDataStoresTest):

    def test_mldataset_levels(self):
        data_store = new_fs_data_store('memory',
                                       root='xcube-test')
        self.assertDatasetSupport(data_store, '.levels',
                                  'dataset[multilevel]',
                                  MultiLevelDataset)

    def test_dataset_zarr(self):
        data_store = new_fs_data_store('memory',
                                       root='xcube-test')
        self.assertDatasetSupport(data_store, '.zarr',
                                  'dataset',
                                  xr.Dataset)

    def test_dataset_netcdf(self):
        data_store = new_fs_data_store('memory',
                                       root='xcube-test')
        self.assertDatasetSupport(data_store, '.nc',
                                  'dataset',
                                  xr.Dataset)


class S3FsDataStoresTest(S3Test, FsDataStoresTest):
    fs_params = dict(
        anon=False,
        client_kwargs=dict(
            endpoint_url=MOTO_SERVER_ENDPOINT_URL,
        )
    )

    def test_mldataset_levels(self):
        data_store = new_fs_data_store('s3',
                                       root='xcube-test',
                                       fs_params=self.fs_params)
        self.assertDatasetSupport(data_store, '.levels',
                                  'dataset[multilevel]',
                                  MultiLevelDataset)

    def test_dataset_zarr(self):
        data_store = new_fs_data_store('s3',
                                       root='xcube-test',
                                       fs_params=self.fs_params)
        self.assertDatasetSupport(data_store, '.zarr',
                                  'dataset',
                                  xr.Dataset)

    def test_dataset_netcdf(self):
        data_store = new_fs_data_store('s3',
                                       root='xcube-test',
                                       fs_params=self.fs_params)
        self.assertDatasetSupport(data_store, '.nc',
                                  'dataset',
                                  xr.Dataset)
