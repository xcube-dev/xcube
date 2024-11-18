# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import xarray

from test.webapi.helpers import get_api_ctx
from xcube.core.mldataset import MultiLevelDataset
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.s3.dsmapping import DatasetsMapping


def get_datasets_ctx(server_config="config-datastores.yml") -> DatasetsContext:
    return get_api_ctx("datasets", DatasetsContext, server_config)


class S3ContextTest(unittest.TestCase):
    def test_datasets_as_zarr(self):
        datasets_ctx = get_datasets_ctx()
        mapping = DatasetsMapping(datasets_ctx, is_multi_level=False)

        expected_s3_names = {"test~cube-1-250-250.zarr", "Cube-T5.zarr", "Cube-T1.zarr"}

        self.assertEqual(len(expected_s3_names), len(mapping))
        self.assertEqual(expected_s3_names, set(mapping))
        for name in expected_s3_names:
            self.assertTrue(name in mapping)
            self.assertIsInstance(mapping[name], xarray.Dataset)

    def test_datasets_as_levels(self):
        datasets_ctx = get_datasets_ctx()
        mapping = DatasetsMapping(datasets_ctx, is_multi_level=True)

        expected_s3_names = {
            "test~cube-1-250-250.levels",
            "Cube-T5.levels",
            "Cube-T1.levels",
        }

        self.assertEqual(len(expected_s3_names), len(mapping))
        self.assertEqual(expected_s3_names, set(mapping))
        for name in expected_s3_names:
            self.assertTrue(name in mapping)
            self.assertIsInstance(mapping[name], MultiLevelDataset)
