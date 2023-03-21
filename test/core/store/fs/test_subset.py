import unittest
from abc import ABC, abstractmethod
from typing import Tuple

import fsspec

from test.s3test import MOTO_SERVER_ENDPOINT_URL
from test.s3test import S3Test
from xcube.core.new import new_cube
from xcube.core.store.fs.registry import new_fs_data_store
from xcube.core.store.fs.store import FsDataStore
from xcube.util.temp import new_temp_dir


class FsStoreSubset:
    """This class hides abstract CommonTest, so pytest will not find it."""

    class CommonTest(unittest.TestCase, ABC):
        fs: fsspec.AbstractFileSystem
        root: str

        @classmethod
        @abstractmethod
        def get_fs_root(cls) -> Tuple[fsspec.AbstractFileSystem, str]:
            pass

        @classmethod
        @abstractmethod
        def new_store(cls, **params) -> FsDataStore:
            pass

        @classmethod
        def setUpClass(cls) -> None:
            super().setUpClass()

            cube = new_cube(width=18, height=9, x_res=20, y_res=20)

            cls.fs, cls.root = cls.get_fs_root()

            cls.fs.delete(cls.root, recursive=True)

            dir_path = f'{cls.root}/l1b'
            cls.fs.mkdir(dir_path)
            for i in range(3):
                zarr_path = f'{dir_path}/olci-l1b-2022050{i + 1}.zarr'
                cube.to_zarr(cls.fs.get_mapper(root=zarr_path), mode='w')

            dir_path = f'{cls.root}/l2'
            cls.fs.mkdir(dir_path)
            for i in range(3):
                zarr_path = f'{dir_path}/olci-l2-2022050{i + 1}.zarr'
                cube.to_zarr(cls.fs.get_mapper(root=zarr_path), mode='w')

            dir_path = f'{cls.root}/l3'
            cls.fs.mkdir(dir_path)
            for i in ['2020', '2021']:
                levels_path = f'{dir_path}/olci-l3-{i}.levels'
                cls.fs.mkdir(levels_path)
                for j in range(4):
                    zarr_path = f'{levels_path}/{j}.zarr'
                    cube.to_zarr(cls.fs.get_mapper(root=zarr_path), mode='w')

        @classmethod
        def tearDownClass(cls) -> None:
            cls.fs.delete(cls.root, recursive=True)
            super().tearDownClass()

        def test_no_subset(self):
            store = self.new_store()
            self.assertEqual({'l1b/olci-l1b-20220501.zarr',
                              'l1b/olci-l1b-20220502.zarr',
                              'l1b/olci-l1b-20220503.zarr',
                              'l2/olci-l2-20220501.zarr',
                              'l2/olci-l2-20220502.zarr',
                              'l2/olci-l2-20220503.zarr',
                              'l3/olci-l3-2020.levels',
                              'l3/olci-l3-2021.levels'},
                             set(store.get_data_ids()))

        def test_include(self):
            store = self.new_store(includes='*/*20220502*')
            self.assertEqual({'l1b/olci-l1b-20220502.zarr',
                              'l2/olci-l2-20220502.zarr'},
                             set(store.get_data_ids()))

            store = self.new_store(includes=['*/*20220502*', '*.levels'])
            self.assertEqual({'l1b/olci-l1b-20220502.zarr',
                              'l2/olci-l2-20220502.zarr',
                              'l3/olci-l3-2020.levels',
                              'l3/olci-l3-2021.levels'},
                             set(store.get_data_ids()))

            store = self.new_store(includes=['*.tiff'])
            self.assertEqual(set(),
                             set(store.get_data_ids()))

        def test_exclude(self):
            store = self.new_store(excludes='*/*20220502*')
            self.assertEqual({'l1b/olci-l1b-20220501.zarr',
                              'l1b/olci-l1b-20220503.zarr',
                              'l2/olci-l2-20220501.zarr',
                              'l2/olci-l2-20220503.zarr',
                              'l3/olci-l3-2020.levels',
                              'l3/olci-l3-2021.levels'},
                             set(store.get_data_ids()))

            store = self.new_store(excludes=['*/*20220502*', '*.levels'])
            self.assertEqual({'l1b/olci-l1b-20220501.zarr',
                              'l1b/olci-l1b-20220503.zarr',
                              'l2/olci-l2-20220501.zarr',
                              'l2/olci-l2-20220503.zarr'},
                             set(store.get_data_ids()))

            store = self.new_store(excludes=['*.zarr', '*.levels'])
            self.assertEqual(set(),
                             set(store.get_data_ids()))

        def test_include_exclude(self):
            store = self.new_store(includes='*.levels',
                                   excludes='*2021*')
            self.assertEqual({'l3/olci-l3-2020.levels'},
                             set(store.get_data_ids()))

            store = self.new_store(includes='*2022*',
                                   excludes=['*.levels', 'l1b/*'])
            self.assertEqual({'l2/olci-l2-20220501.zarr',
                              'l2/olci-l2-20220502.zarr',
                              'l2/olci-l2-20220503.zarr'},
                             set(store.get_data_ids()))


class MemoryFsStoreSubsetTest(FsStoreSubset.CommonTest):
    @classmethod
    def get_fs_root(cls) -> Tuple[fsspec.AbstractFileSystem, str]:
        fs = fsspec.get_filesystem_class("memory")()
        root = 'xcube'
        fs.mkdirs(root, exist_ok=True)
        return fs, root

    @classmethod
    def new_store(cls, **params) -> FsDataStore:
        return new_fs_data_store('memory',
                                 root=cls.root,
                                 max_depth=3,
                                 read_only=True,
                                 **params)


class FileFsStoreSubsetTest(FsStoreSubset.CommonTest):

    @classmethod
    def get_fs_root(cls) -> Tuple[fsspec.AbstractFileSystem, str]:
        return fsspec.get_filesystem_class("file")(), new_temp_dir()

    @classmethod
    def new_store(cls, **params) -> FsDataStore:
        return new_fs_data_store('file',
                                 root=cls.root,
                                 max_depth=3,
                                 read_only=True,
                                 **params)


# TODO (forman): check, why we get OSError for moto test
@unittest.skip("OSError: [Errno 5] Internal Server Error")
class S3FsStoreSubsetTest(FsStoreSubset.CommonTest, S3Test):
    @classmethod
    def get_fs_root(cls) -> Tuple[fsspec.AbstractFileSystem, str]:
        fs = fsspec.get_filesystem_class("s3")(**cls.get_storage_options())
        root = 'xcube'
        fs.mkdirs(root, exist_ok=True)
        return fs, root

    @classmethod
    def new_store(cls, **params) -> FsDataStore:
        return new_fs_data_store(
            's3',
            root=cls.root,
            max_depth=3,
            read_only=True,
            storage_options=cls.get_storage_options(),
            **params
        )

    @classmethod
    def get_storage_options(cls):
        return dict(
            anon=True,
            key="",
            secret="",
            client_kwargs=dict(
                endpoint_url=MOTO_SERVER_ENDPOINT_URL,
            )
        )
