# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
import unittest

from fsspec.registry import register_implementation

from xcube.core.store import DataStoreError
from xcube.core.store import list_data_store_ids
from xcube.core.store import new_data_store
import pytest


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


def test_fsspec_instantiation_error():
    error_string = "deliberate instantiation error for testing"
    register_implementation(
        "abfs", "nonexistentmodule.NonexistentClass", True, error_string
    )
    with pytest.raises(DataStoreError, match=error_string):
        new_data_store("abfs").list_data_ids()
