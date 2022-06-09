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
        def create_store(cls, **params) -> FsDataStore:
            pass

        @classmethod
        def setUpClass(cls) -> None:
            super().setUpClass()

            cube = new_cube(width=18, height=9, x_res=20, y_res=20)

            cls.fs, cls.root = cls.get_fs_root()
            zarr_store = cls.fs.get_mapper(root=cls.root)

            cls.fs.mkdir('l1b')
            for i in range(3):
                cube.to_zarr(zarr_store,
                             f'l1b/olci-l1b-2022050{i + 1}.zarr',
                             mode='w')

            cls.fs.mkdir('l2')
            for i in range(3):
                cube.to_zarr(zarr_store,
                             f'l2/olci-l2-2022050{i + 1}.zarr',
                             mode='w')

            cls.fs.mkdir('l3')
            for i in ['2020', '2021']:
                path = f'l3/olci-l3-{i}.levels'
                cls.fs.mkdir(path)
                for j in range(4):
                    cube.to_zarr(zarr_store,
                                 f'l3/{path}/{j}.zarr',
                                 mode='w')

        @classmethod
        def tearDownClass(cls) -> None:
            cls.fs.delete(cls.root, recursive=True)
            super().tearDownClass()

        def test_no_subset(self):
            store = self.create_store()
            self.assertEqual(['l1b/olci-l1b-20220501.zarr',
                              'l1b/olci-l1b-20220502.zarr',
                              'l1b/olci-l1b-20220503.zarr',
                              'l2/olci-l2-20220501.zarr',
                              'l2/olci-l2-20220502.zarr',
                              'l2/olci-l2-20220503.zarr',
                              'l3/olci-l3-2020.levels',
                              'l3/olci-l3-2021.levels'],
                             list(store.get_data_ids()))


class MemoryFsStoreSubsetTest(FsStoreSubset.CommonTest):
    @classmethod
    def get_fs_root(cls) -> Tuple[fsspec.AbstractFileSystem, str]:
        fs = fsspec.get_filesystem_class("memory")()
        return fs, 'xcube'

    @classmethod
    def create_store(cls, **params) -> FsDataStore:
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
    def create_store(cls, **params) -> FsDataStore:
        return new_fs_data_store('file',
                                 root=cls.root,
                                 max_depth=3,
                                 read_only=True,
                                 **params)


class S3FsStoreSubsetTest(FsStoreSubset.CommonTest, S3Test):
    @classmethod
    def get_fs_root(cls) -> Tuple[fsspec.AbstractFileSystem, str]:
        fs = fsspec.get_filesystem_class("s3")()
        fs.mkdir('xcube')
        return fs, 'xcube'

    @classmethod
    def create_store(cls, **params) -> FsDataStore:
        return new_fs_data_store('s3',
                                 root=cls.root,
                                 max_depth=3,
                                 storage_options=cls.storage_options(),
                                 read_only=True,
                                 **params)

    @classmethod
    def storage_options(cls):
        storage_options = dict(
            anon=False,
            client_kwargs=dict(
                endpoint_url=MOTO_SERVER_ENDPOINT_URL,
            )
        )
        return storage_options
