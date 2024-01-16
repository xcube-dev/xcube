import collections.abc
import os.path
import shutil
import unittest
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Set, Type, Union

import fsspec
import numpy as np
import pytest
import xarray as xr

import xcube.core.mldataset
from test.s3test import MOTO_SERVER_ENDPOINT_URL
from test.s3test import S3Test
from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.new import new_cube
from xcube.core.store import DataDescriptor
from xcube.core.store import DataStoreError
from xcube.core.store import DatasetDescriptor
from xcube.core.store import MultiLevelDatasetDescriptor
from xcube.core.store import MutableDataStore
from xcube.core.store.fs.registry import new_fs_data_store
from xcube.core.store.fs.store import FsDataStore
from xcube.core.zarrstore import GenericZarrStore
from xcube.util.temp import new_temp_dir
from xcube.constants import CRS84

ROOT_DIR = 'xcube'
DATA_PATH = 'testing/data'


def new_cube_data():
    width = 360
    height = 180
    time_periods = 5
    shape = (time_periods, height, width)
    var_a = np.full(shape, 8.5, dtype=np.float64)
    var_b = np.full(shape, 9.5, dtype=np.float64)
    var_c = np.full(shape, 255, dtype=np.uint8)

    var_a[0, 0, 0] = np.nan
    var_b[0, 0, 0] = np.nan

    cube = new_cube(width=width,
                    height=height,
                    x_name='x',
                    y_name='y',
                    crs=CRS84,
                    crs_name='spatial_ref',
                    time_periods=time_periods,
                    variables=dict(var_a=var_a,
                                   var_b=var_b,
                                   var_c=var_c))

    # Set var_b encodings
    cube.var_b.encoding['dtype'] = np.int16
    cube.var_b.encoding['_FillValue'] = -9999
    cube.var_b.encoding['scale_factor'] = 0.001
    cube.var_b.encoding['add_offset'] = -10

    return cube.chunk(dict(time=1, y=90, x=180))


class NewCubeDataTestMixin(unittest.TestCase):
    path = f'{DATA_PATH}/data.zarr'

    @classmethod
    def setUpClass(cls) -> None:
        data = new_cube_data()
        data.to_zarr(cls.path, mode="w")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.path)

    def test_open_unpacked(self):
        """open data un-packed (the default)"""
        data_1 = xr.open_zarr(self.path, mask_and_scale=True)
        self.assertEqual(np.float64, data_1.var_a.dtype)
        self.assertEqual(np.float32, data_1.var_b.dtype)
        self.assertEqual(np.uint8, data_1.var_c.dtype)
        self.assertTrue(np.isnan(data_1.var_a[0, 0, 0]))
        self.assertEqual(8.5, data_1.var_a[1, 0, 0].values)
        self.assertTrue(np.isnan(data_1.var_b[0, 0, 0]))
        self.assertEqual(9.5, data_1.var_b[1, 0, 0].values)
        self.assertEqual(255, data_1.var_c[0, 0, 0].values)
        self.assertEqual(255, data_1.var_c[1, 0, 0].values)

    def test_open_packed(self):
        """open data packed, ignoring related encodings"""
        data_2 = xr.open_zarr(self.path, mask_and_scale=False)
        self.assertEqual(np.float64, data_2.var_a.dtype)
        self.assertEqual(np.int16, data_2.var_b.dtype)
        self.assertEqual(np.uint8, data_2.var_c.dtype)
        self.assertTrue(np.isnan(data_2.var_a[0, 0, 0]))
        self.assertEqual(8.5, data_2.var_a[1, 0, 0].values)
        self.assertEqual(-9999, data_2.var_b[0, 0, 0].values)
        self.assertEqual((9.5 - (-10)) / 0.001, data_2.var_b[1, 0, 0].values)
        self.assertEqual(255, data_2.var_c[0, 0, 0].values)
        self.assertEqual(255, data_2.var_c[1, 0, 0].values)


# noinspection PyUnresolvedReferences,PyPep8Naming
class FsDataStoresTestMixin(ABC):
    @abstractmethod
    def create_data_store(self) -> FsDataStore:
        pass

    @classmethod
    def prepare_fs(cls, fs: fsspec.AbstractFileSystem, root: str):
        if fs.isdir(root):
            # print(f'{fs.protocol}: deleting {root}')
            fs.delete(root, recursive=True)

        # print(f'{fs.protocol}: making root {root}')
        fs.mkdirs(root)

        # Write a text file into each subdirectory, so
        # we also test that store.get_data_ids() scans
        # recursively.
        dir_path = root
        for subdir_name in DATA_PATH.split('/'):
            dir_path += '/' + subdir_name
            # print(f'{fs.protocol}: making {dir_path}')
            fs.mkdir(dir_path)
            file_path = dir_path + '/README.md'
            # print(f'{fs.protocol}: writing {file_path}')
            with fs.open(file_path, 'w') as fp:
                fp.write('\n')

    def test_mldataset_levels(self):
        data_store = self.create_data_store()
        self._assert_multi_level_dataset_format_supported(data_store)
        self._assert_multi_level_dataset_format_with_link_supported(
            data_store)
        self._assert_multi_level_dataset_format_with_tile_size(data_store)

    def test_dataset_zarr(self):
        data_store = self.create_data_store()
        self._assert_dataset_supported(
            data_store,
            filename_ext='.zarr',
            requested_dtype_alias=None,
            expected_dtype_aliases={'dataset'},
            expected_return_type=xr.Dataset,
            expected_descriptor_type=DatasetDescriptor,
            assert_data_ok=self._assert_zarr_store_direct_ok
        )

    def test_dataset_netcdf(self):
        data_store = self.create_data_store()
        self._assert_dataset_supported(
            data_store,
            filename_ext='.nc',
            requested_dtype_alias=None,
            expected_dtype_aliases={'dataset'},
            expected_return_type=xr.Dataset,
            expected_descriptor_type=DatasetDescriptor,
            assert_data_ok=self._assert_zarr_store_generic_ok
        )

    def test_dataset_levels(self):
        data_store = self.create_data_store()
        self._assert_dataset_supported(
            data_store,
            filename_ext='.levels',
            requested_dtype_alias='dataset',
            expected_dtype_aliases={'mldataset', 'dataset'},
            expected_return_type=xr.Dataset,
            expected_descriptor_type=None,
            assert_data_ok=self._assert_zarr_store_direct_ok
        )

    # TODO: add assertGeoDataFrameSupport

    def _assert_multi_level_dataset_format_supported(
            self,
            data_store: FsDataStore
    ):
        self._assert_dataset_supported(
            data_store,
            filename_ext='.levels',
            requested_dtype_alias=None,
            expected_dtype_aliases={'mldataset', 'dataset'},
            expected_return_type=MultiLevelDataset,
            expected_descriptor_type=MultiLevelDatasetDescriptor,
            assert_data_ok=self._assert_multi_level_dataset_data_ok
        )

        # Test that use_saved_levels works
        self._assert_dataset_supported(
            data_store,
            filename_ext='.levels',
            requested_dtype_alias=None,
            expected_dtype_aliases={'mldataset', 'dataset'},
            expected_return_type=MultiLevelDataset,
            expected_descriptor_type=MultiLevelDatasetDescriptor,
            write_params=dict(
                use_saved_levels=True,
            ),
            assert_data_ok=self._assert_multi_level_dataset_data_ok
        )

    def _assert_multi_level_dataset_format_with_link_supported(
            self,
            data_store: FsDataStore
    ):
        base_dataset = new_cube_data()
        base_dataset_id = f'{DATA_PATH}/base-ds.zarr'
        data_store.write_data(base_dataset, base_dataset_id)

        # Test that base_dataset_id works
        self._assert_dataset_supported(
            data_store,
            filename_ext='.levels',
            requested_dtype_alias=None,
            expected_dtype_aliases={'mldataset', 'dataset'},
            expected_return_type=MultiLevelDataset,
            expected_descriptor_type=MultiLevelDatasetDescriptor,
            write_params=dict(
                base_dataset_id=base_dataset_id,
            ),
            assert_data_ok=self._assert_multi_level_dataset_data_ok
        )

        # Test that base_dataset_id + use_saved_levels works
        self._assert_dataset_supported(
            data_store,
            filename_ext='.levels',
            requested_dtype_alias=None,
            expected_dtype_aliases={'mldataset', 'dataset'},
            expected_return_type=MultiLevelDataset,
            expected_descriptor_type=MultiLevelDatasetDescriptor,
            write_params=dict(
                base_dataset_id=base_dataset_id,
                use_saved_levels=True,
            ),
            assert_data_ok=self._assert_multi_level_dataset_data_ok
        )

        data_store.delete_data(base_dataset_id)

    def _assert_zarr_store_direct_ok(self, dataset):
        self.assertIsInstance(dataset, xr.Dataset)
        self.assertTrue(hasattr(dataset, 'zarr_store'))
        self.assertIsInstance(dataset.zarr_store.get(),
                              collections.abc.MutableMapping)
        self.assertNotIsInstance(dataset.zarr_store.get(),
                                 GenericZarrStore)

    def _assert_zarr_store_generic_ok(self, dataset):
        self.assertIsInstance(dataset, xr.Dataset)
        self.assertTrue(hasattr(dataset, 'zarr_store'))
        self.assertIsInstance(dataset.zarr_store.get(),
                              GenericZarrStore)

    def _assert_multi_level_dataset_data_ok(self, ml_dataset):
        self.assertIsInstance(ml_dataset,
                              xcube.core.mldataset.MultiLevelDataset)
        self.assertEqual(2, ml_dataset.num_levels)
        self.assertIsInstance(ml_dataset.grid_mapping, GridMapping)
        self.assertIsInstance(ml_dataset.base_dataset, xr.Dataset)
        self.assertIsInstance(ml_dataset.ds_id, str)
        # assert encoding
        for level in range(ml_dataset.num_levels):
            dataset = ml_dataset.get_dataset(level)
            self.assertEqual({'var_a',
                              'var_b',
                              'var_c',
                              'spatial_ref'},
                             set(dataset.data_vars))
            # assert dtype is as expected
            self.assertEqual(np.float64, dataset.var_a.dtype)
            self.assertEqual(np.float32, dataset.var_b.dtype)
            self.assertEqual(np.uint8, dataset.var_c.dtype)
            # assert dtype encoding is as expected
            self.assertEqual(np.float64,
                             dataset.var_a.encoding.get('dtype'))
            self.assertEqual(np.int16,
                             dataset.var_b.encoding.get('dtype'))
            self.assertEqual(np.uint8,
                             dataset.var_c.encoding.get('dtype'))
            # assert _FillValue encoding is as expected
            self.assertTrue(np.isnan(
                dataset.var_a.encoding.get('_FillValue')
            ))
            self.assertEqual(-9999,
                             dataset.var_b.encoding.get('_FillValue'))
            self.assertEqual(None,
                             dataset.var_c.encoding.get('_FillValue'))

            self.assertTrue(hasattr(dataset, 'zarr_store'))
            self.assertIsInstance(dataset.zarr_store.get(),
                                  collections.abc.MutableMapping)
            self.assertNotIsInstance(dataset.zarr_store.get(),
                                     GenericZarrStore)

    def _assert_multi_level_dataset_format_with_tile_size(
            self,
            data_store: FsDataStore
    ):
        base_dataset = new_cube_data()
        base_dataset_id = f'{DATA_PATH}/base-ds.zarr'
        data_store.write_data(base_dataset, base_dataset_id)

        # Test that base_dataset_id works
        self._assert_dataset_supported(
            data_store,
            filename_ext='.levels',
            requested_dtype_alias=None,
            expected_dtype_aliases={'mldataset', 'dataset'},
            expected_return_type=MultiLevelDataset,
            expected_descriptor_type=MultiLevelDatasetDescriptor,
            open_params=dict(cache_size=2 ** 20),
            write_params=dict(tile_size=90)
        )

        # Test that base_dataset_id + use_saved_levels works
        self._assert_dataset_supported(
            data_store,
            filename_ext='.levels',
            requested_dtype_alias=None,
            expected_dtype_aliases={'mldataset', 'dataset'},
            expected_return_type=MultiLevelDataset,
            expected_descriptor_type=MultiLevelDatasetDescriptor,
            open_params=dict(cache_size=2 ** 20),
            write_params=dict(
                tile_size=90,
                use_saved_levels=True,
            )
        )

        data_store.delete_data(base_dataset_id)

    def _assert_dataset_supported(
            self,
            data_store: FsDataStore,
            filename_ext: str,
            requested_dtype_alias: Optional[str],
            expected_dtype_aliases: Set[str],
            expected_return_type: Union[Type[xr.Dataset],
                                        Type[MultiLevelDataset]],
            expected_descriptor_type: Optional[Union[
                Type[DatasetDescriptor],
                Type[MultiLevelDatasetDescriptor]
            ]] = None,
            write_params: Optional[Dict[str, Any]] = None,
            open_params: Optional[Dict[str, Any]] = None,
            assert_data_ok: Optional[Callable[[Any], Any]] = None
    ):
        """
        Call all DataStore operations to ensure data of type
        xr.Dataset//MultiLevelDataset is supported by *data_store*.

        :param data_store: The filesystem data store instance.
        :param filename_ext: Filename extension that identifies
            a supported dataset format.
        :param expected_data_type_alias: The expected data type alias.
        :param expected_return_type: The expected data type.
        :param expected_descriptor_type: The expected data descriptor type.
        :param write_params: Optional write parameters
        :param open_params: Optional open parameters
        :param assert_data_ok: Optional function to assert read data is ok
        """

        data_id = f'{DATA_PATH}/ds{filename_ext}'

        write_params = write_params or {}
        open_params = open_params or {}

        self.assertIsInstance(data_store, MutableDataStore)

        self.assertEqual({'dataset', 'mldataset', 'geodataframe'},
                         set(data_store.get_data_types()))

        with pytest.raises(DataStoreError,
                           match=f'Data resource "{data_id}"'
                                 f' does not exist in store'):
            data_store.get_data_types_for_data(data_id)
        self.assertEqual(False, data_store.has_data(data_id))
        self.assertNotIn(data_id, set(data_store.get_data_ids()))
        self.assertNotIn(data_id, data_store.list_data_ids())

        data = new_cube_data()
        written_data_id = data_store.write_data(data, data_id, **write_params)
        self.assertEqual(data_id, written_data_id)

        self.assertEqual(expected_dtype_aliases,
                         set(data_store.get_data_types_for_data(data_id)))
        self.assertEqual(True, data_store.has_data(data_id))
        self.assertIn(data_id, set(data_store.get_data_ids()))
        self.assertIn(data_id, data_store.list_data_ids())

        if expected_descriptor_type is not None:
            data_descriptors = list(data_store.search_data(
                data_type=expected_return_type)
            )
            self.assertEqual(1, len(data_descriptors))
            self.assertIsInstance(data_descriptors[0],
                                  DataDescriptor)
            self.assertIsInstance(data_descriptors[0],
                                  expected_descriptor_type)

        if requested_dtype_alias:
            # noinspection PyProtectedMember
            _, format_name, protocol = \
                data_store._guess_accessor_id_parts(data_id)
            opener_id = f'{requested_dtype_alias}:{format_name}:{protocol}'
        else:
            opener_id = None
        data = data_store.open_data(data_id,
                                    opener_id=opener_id,
                                    **open_params)
        self.assertIsInstance(data, expected_return_type)
        if assert_data_ok is not None:
            assert_data_ok(data)

        try:
            data_store.delete_data(data_id)
        except PermissionError as e:  # May occur on win32 due to fsspec
            warnings.warn(f'{e}')
            return
        with pytest.raises(DataStoreError,
                           match=f'Data resource "{data_id}"'
                                 f' does not exist in store'):
            data_store.get_data_types_for_data(data_id)
        self.assertEqual(False, data_store.has_data(data_id))
        self.assertNotIn(data_id, set(data_store.get_data_ids()))
        self.assertNotIn(data_id, data_store.list_data_ids())


class FileFsDataStoresTest(FsDataStoresTestMixin, unittest.TestCase):
    def create_data_store(self) -> FsDataStore:
        root = os.path.join(new_temp_dir(prefix='xcube'), ROOT_DIR)
        self.prepare_fs(fsspec.filesystem('file'), root)
        return new_fs_data_store('file', root=root, max_depth=3)


class MemoryFsDataStoresTest(FsDataStoresTestMixin, unittest.TestCase):

    def create_data_store(self) -> FsDataStore:
        root = ROOT_DIR
        self.prepare_fs(fsspec.filesystem('memory'), root)
        return new_fs_data_store('memory', root=root, max_depth=3)


class S3FsDataStoresTest(FsDataStoresTestMixin, S3Test):

    def create_data_store(self) -> FsDataStore:
        root = ROOT_DIR
        storage_options = dict(
            anon=False,
            client_kwargs=dict(
                endpoint_url=MOTO_SERVER_ENDPOINT_URL,
            )
        )
        self.prepare_fs(fsspec.filesystem('s3', **storage_options), root)
        return new_fs_data_store('s3',
                                 root=root,
                                 max_depth=3,
                                 storage_options=storage_options)
