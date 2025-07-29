# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
from typing import Literal
import unittest
from unittest.mock import MagicMock, patch

import pytest
from fsspec.registry import register_implementation

from xcube.core.store import (
    DataStoreError,
    get_data_store_class,
    list_data_store_ids,
    new_data_store,
)
from xcube.core.store.fs.store import BaseFsDataStore
from xcube.core.store.preload import NullPreloadHandle


class ListDataStoreTest(unittest.TestCase):
    def test_list_data_store_ids(self):
        ids = list_data_store_ids()
        self.assertIsInstance(ids, list)
        self.assertIn("file", ids)
        self.assertIn("s3", ids)
        self.assertIn("memory", ids)
        self.assertIn("ftp", ids)
        self.assertIn("reference", ids)

    def test_list_data_store_ids_detail(self):
        ids = list_data_store_ids(detail=True)
        self.assertIsInstance(ids, dict)
        self.assertEqual(
            {"description": "Data store that uses a local filesystem"}, ids.get("file")
        )
        self.assertEqual(
            {"description": "Data store that uses a AWS S3 compatible object storage"},
            ids.get("s3"),
        )
        self.assertEqual(
            {"description": "Data store that uses a in-memory filesystem"},
            ids.get("memory"),
        )
        self.assertEqual(
            {"description": "Data store that uses a FTP filesystem"},
            ids.get("ftp"),
        )
        self.assertEqual(
            {"description": "Data store that uses Kerchunk references"},
            ids.get("reference"),
        )


class TestBaseFsDataStore(unittest.TestCase):

    def test_get_data_types(self):
        self.assertEqual(
            {"dataset", "geodataframe", "mldataset"},
            set(BaseFsDataStore.get_data_types())
        )

    def test_get_data_opener_ids(self):
        store = new_data_store("file")
        self.assertEqual(
            ("mldataset:geotiff:file", "dataset:geotiff:file",),
            store.get_data_opener_ids(data_id="test.geotiff")
        )
        self.assertEqual(
            ("mldataset:geotiff:file",),
            store.get_data_opener_ids(data_id="test.geotiff", data_type="mldataset"),
        )

    @patch("fsspec.filesystem")
    def test_has_data(self, mock_filesystem):
        # Mock the HTTPFileSystem instance and its `exists` method
        mock_http_fs = MagicMock()
        mock_filesystem.return_value = mock_http_fs
        mock_http_fs.exists.return_value = True
        mock_http_fs.sep = "/"

        store = new_data_store("https", root="test.org")

        res = store.has_data(data_id="test.tif")
        self.assertEqual(mock_filesystem.call_count, 1)
        mock_http_fs.exists.assert_called_once_with("https://test.org/test.tif")
        self.assertTrue(res)

        res = store.has_data(data_id="test.tif", data_type="dataset")
        mock_http_fs.exists.assert_called_with("https://test.org/test.tif")
        self.assertEqual(mock_http_fs.exists.call_count, 2)
        self.assertTrue(res)

        res = store.has_data(data_id="test.tif", data_type="mldataset")
        mock_http_fs.exists.assert_called_with("https://test.org/test.tif")
        self.assertEqual(mock_http_fs.exists.call_count, 3)
        self.assertTrue(res)

        res = store.has_data(data_id="test.tif", data_type="geodataframe")
        self.assertEqual(mock_http_fs.exists.call_count, 3)
        self.assertFalse(res)

    def test_preload_data(self):
        store = new_data_store("file")
        store_test = store.preload_data()
        self.assertTrue(hasattr(store_test, "preload_handle"))
        self.assertIsInstance(store_test.preload_handle, NullPreloadHandle)


class FsDataStoreTest(unittest.TestCase):

    def test_get_filename_extensions_abfs_openers(self):
        self.assert_accessors("abfs", "openers")

    def test_get_filename_extensions_abfs_writers(self):
        self.assert_accessors("abfs", "writers")

    def test_get_filename_extensions_file_openers(self):
        self.assert_accessors("file", "openers")

    def test_get_filename_extensions_file_writers(self):
        self.assert_accessors("file", "writers")

    def test_get_filename_extensions_ftp_openers(self):
        self.assert_accessors("ftp", "openers")

    def test_get_filename_extensions_ftp_writers(self):
        self.assert_accessors("ftp", "writers")

    def test_get_filename_extensions_https_openers(self):
        self.assert_accessors("https", "openers")

    def test_get_filename_extensions_https_writers(self):
        self.assert_accessors("https", "writers")

    def test_get_filename_extensions_memory_openers(self):
        self.assert_accessors("memory", "openers")

    def test_get_filename_extensions_memory_writers(self):
        self.assert_accessors("memory", "writers")

    def test_get_filename_extensions_s3_openers(self):
        self.assert_accessors("s3", "openers")

    def test_get_filename_extensions_s3_writers(self):
        self.assert_accessors("s3", "writers")

    def test_get_filename_extensions_unknown_accessor_type(self):
        with self.assertRaises(DataStoreError) as dse:
            self.assert_accessors("s3", "modifiers")
        self.assertEqual(
            "Invalid accessor type. Must be 'openers' or 'writers', was 'modifiers'",
            f"{dse.exception}"
        )

    def assert_accessors(
        self, protocol: str, accessor_type: Literal["openers", "writers"]
    ):
        store_cls = get_data_store_class(protocol)
        accessors = store_cls.get_filename_extensions(accessor_type)
        expected_accessors = {
            '.geojson': [f'geodataframe:geojson:{protocol}'],
            '.levels': [f'mldataset:levels:{protocol}',
                        f'dataset:levels:{protocol}'],
            '.nc': [f'dataset:netcdf:{protocol}'],
            '.shp': [f'geodataframe:shapefile:{protocol}'],
            '.zarr': [f'dataset:zarr:{protocol}']
        }
        if accessor_type == "openers":
            geotiff_openers = {
                '.geotiff': [f'mldataset:geotiff:{protocol}',
                             f'dataset:geotiff:{protocol}'],
                '.tif': [f'mldataset:geotiff:{protocol}',
                         f'dataset:geotiff:{protocol}'],
                '.tiff': [f'mldataset:geotiff:{protocol}',
                      f'dataset:geotiff:{protocol}'],
            }
            expected_accessors.update(geotiff_openers)
        self.assertEqual(accessors, expected_accessors)


def test_fsspec_instantiation_error():
    error_string = "deliberate instantiation error for testing"
    register_implementation(
        "abfs", "nonexistentmodule.NonexistentClass", True, error_string
    )
    with pytest.raises(DataStoreError, match=error_string):
        new_data_store("abfs").list_data_ids()
