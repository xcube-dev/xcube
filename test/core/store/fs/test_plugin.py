# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.core.store import DataOpener
from xcube.core.store import DataWriter
from xcube.core.store import MutableDataStore
from xcube.core.store import find_data_opener_extensions
from xcube.core.store import find_data_store_extensions
from xcube.core.store import find_data_writer_extensions
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

expected_fs_data_accessor_ids: set = {
    "dataset:netcdf:file",
    "dataset:netcdf:memory",
    "dataset:netcdf:s3",
    "dataset:zarr:file",
    "dataset:zarr:memory",
    "dataset:zarr:s3",
    "mldataset:levels:file",
    "mldataset:levels:memory",
    "mldataset:levels:s3",
    "geodataframe:geojson:file",
    "geodataframe:geojson:memory",
    "geodataframe:geojson:s3",
    "geodataframe:shapefile:file",
    "geodataframe:shapefile:memory",
    "geodataframe:shapefile:s3",
}

expected_fs_store_ids: set = {
    "file",
    "memory",
    "s3",
}


class FsDataStoreAndAccessorsPluginTest(unittest.TestCase):
    """
    Make sure all expected Filesystem data stores, openers, and writers
    are correctly registered.
    """

    def test_find_data_store_extensions(self):
        extensions = find_data_store_extensions()
        self.assertTrue(len(extensions) >= len(expected_fs_store_ids))
        self.assertEqual({"xcube.core.store"}, {ext.point for ext in extensions})
        self.assertTrue(
            expected_fs_store_ids.issubset({ext.name for ext in extensions})
        )
        for ext in extensions:
            if ext.name not in expected_fs_store_ids:
                continue
            data_store_class = ext.component
            self.assertTrue(callable(data_store_class))
            data_store = data_store_class()
            self.assertIsInstance(data_store, MutableDataStore)
            params_schema = data_store.get_data_store_params_schema()
            self.assertParamsSchemaIncludesFsParams(params_schema)
            params_schema = data_store.get_open_data_params_schema()
            self.assertParamsSchemaIncludesDataTypeParam(params_schema)
            self.assertParamsSchemaExcludesFsParams(params_schema)
            params_schema = data_store.get_delete_data_params_schema()
            self.assertParamsSchemaExcludesFsParams(params_schema)
            self.assertTrue(len(data_store.get_data_opener_ids()) >= 4)
            self.assertTrue(len(data_store.get_data_writer_ids()) >= 4)

    def test_find_data_opener_extensions(self):
        extensions = find_data_opener_extensions()
        self.assertTrue(len(extensions) >= len(expected_fs_data_accessor_ids))
        self.assertEqual({"xcube.core.store.opener"}, {ext.point for ext in extensions})
        self.assertTrue(
            expected_fs_data_accessor_ids.issubset({ext.name for ext in extensions})
        )
        for ext in extensions:
            if ext.name not in expected_fs_data_accessor_ids:
                continue
            data_opener_class = ext.component
            self.assertTrue(callable(data_opener_class))
            data_opener = data_opener_class()
            self.assertIsInstance(data_opener, DataOpener)
            params_schema = data_opener.get_open_data_params_schema()
            self.assertParamsSchemaIncludesFsParams(params_schema)

    def test_find_data_writer_extensions(self):
        extensions = find_data_writer_extensions()
        self.assertTrue(len(extensions) >= len(expected_fs_data_accessor_ids))
        self.assertEqual({"xcube.core.store.writer"}, {ext.point for ext in extensions})
        self.assertTrue(
            expected_fs_data_accessor_ids.issubset({ext.name for ext in extensions})
        )
        for ext in extensions:
            if ext.name not in expected_fs_data_accessor_ids:
                continue
            data_writer_class = ext.component
            self.assertTrue(callable(data_writer_class))
            data_writer = data_writer_class()
            self.assertIsInstance(data_writer, DataWriter)
            params_schema = data_writer.get_write_data_params_schema()
            self.assertParamsSchemaIncludesFsParams(params_schema)
            params_schema = data_writer.get_delete_data_params_schema()
            self.assertParamsSchemaIncludesFsParams(params_schema)

    def assertParamsSchemaIncludesDataTypeParam(self, params_schema):
        # print(params_schema.to_dict())
        self.assertIsInstance(params_schema, JsonObjectSchema)
        self.assertIsInstance(params_schema.properties, dict)
        self.assertIn("data_type", params_schema.properties)
        self.assertIsInstance(params_schema.properties["data_type"], JsonStringSchema)

    def assertParamsSchemaIncludesFsParams(self, params_schema):
        # print(params_schema.to_dict())
        self.assertIsInstance(params_schema, JsonObjectSchema)
        self.assertIsInstance(params_schema.properties, dict)
        self.assertIn("storage_options", params_schema.properties)
        self.assertIsInstance(
            params_schema.properties["storage_options"], JsonObjectSchema
        )

    def assertParamsSchemaExcludesFsParams(self, params_schema):
        # print(params_schema.to_dict())
        self.assertIsInstance(params_schema, JsonObjectSchema)
        self.assertIsInstance(params_schema.properties, dict)
        self.assertNotIn("storage_options", params_schema.properties)
