# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
import unittest
from unittest.mock import MagicMock, patch

import pytest
from fsspec.registry import register_implementation

from xcube.core.store import DataStoreError, list_data_store_ids, new_data_store


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
    def test_get_data_opener_ids(self):
        store = new_data_store("file")
        self.assertEqual(
            ("dataset:geotiff:file",), store.get_data_opener_ids(data_id="test.geotiff")
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


def test_fsspec_instantiation_error():
    error_string = "deliberate instantiation error for testing"
    register_implementation(
        "abfs", "nonexistentmodule.NonexistentClass", True, error_string
    )
    with pytest.raises(DataStoreError, match=error_string):
        new_data_store("abfs").list_data_ids()
