# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import unittest

import xarray

from test.webapi.helpers import get_api_ctx
from xcube.core.mldataset import MultiLevelDataset
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.s3.dsmapping import DatasetsMapping


def get_datasets_ctx(
        server_config='config-datastores.yml'
) -> DatasetsContext:
    return get_api_ctx("datasets", DatasetsContext, server_config)


class S3ContextTest(unittest.TestCase):

    def test_datasets_as_zarr(self):
        datasets_ctx = get_datasets_ctx()
        mapping = DatasetsMapping(datasets_ctx, is_multi_level=False)

        expected_s3_names = {'test~cube-1-250-250.zarr',
                             'Cube-T5.zarr',
                             'Cube-T1.zarr'}

        self.assertEqual(len(expected_s3_names), len(mapping))
        self.assertEqual(expected_s3_names, set(mapping))
        for name in expected_s3_names:
            self.assertTrue(name in mapping)
            self.assertIsInstance(mapping[name], xarray.Dataset)

    def test_datasets_as_levels(self):
        datasets_ctx = get_datasets_ctx()
        mapping = DatasetsMapping(datasets_ctx, is_multi_level=True)

        expected_s3_names = {'test~cube-1-250-250.levels',
                             'Cube-T5.levels',
                             'Cube-T1.levels'}

        self.assertEqual(len(expected_s3_names), len(mapping))
        self.assertEqual(expected_s3_names, set(mapping))
        for name in expected_s3_names:
            self.assertTrue(name in mapping)
            self.assertIsInstance(mapping[name], MultiLevelDataset)
