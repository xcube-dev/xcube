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
from typing import Union

from test.webapi.helpers import get_api_ctx
from xcube.server.api import ServerConfig
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.s3.context import S3Context
from xcube.webapi.s3.objectstorage import ObjectStorage


def get_s3_ctx(
        server_config: Union[str, ServerConfig] = "config.yml"
) -> S3Context:
    return get_api_ctx("s3", S3Context, server_config)


class S3ContextTest(unittest.TestCase):

    def test_ctx_ok(self):
        ctx = get_s3_ctx()
        self.assertIsInstance(ctx.datasets_ctx, DatasetsContext)
        self.assertIsInstance(ctx.get_bucket("datasets"),
                              ObjectStorage)
        self.assertIsInstance(ctx.get_bucket("pyramids"),
                              ObjectStorage)

    def test_datasets_bucket(self):
        ctx = get_s3_ctx()
        bucket = ctx.get_bucket("datasets")
        self.assertEqual(['demo.zarr/.zmetadata',
                          'demo.zarr/.zgroup',
                          'demo.zarr/.zattrs',
                          'demo.zarr/lat/.zarray',
                          'demo.zarr/lat/.zattrs',
                          'demo.zarr/lat/0',
                          'demo.zarr/lat_bnds/.zarray',
                          'demo.zarr/lat_bnds/.zattrs',
                          'demo.zarr/lat_bnds/0.0',
                          'demo.zarr/lon/.zarray'],
                         list(bucket.keys())[0:10])

        self.assertTrue('demo.zarr/lat/.zarray' in bucket)
        self.assertTrue('demo.zarr/lat_bnds/0.0' in bucket)
        self.assertTrue('demi.zarr/lat_bnds/0.0' not in bucket)

    def test_pyramids_bucket(self):
        ctx = get_s3_ctx()
        bucket = ctx.get_bucket("pyramids")
        self.assertEqual(['demo.levels/0.zarr/.zmetadata',
                          'demo.levels/0.zarr/.zgroup',
                          'demo.levels/0.zarr/.zattrs',
                          'demo.levels/0.zarr/lat/.zarray',
                          'demo.levels/0.zarr/lat/.zattrs',
                          'demo.levels/0.zarr/lat/0',
                          'demo.levels/0.zarr/lat_bnds/.zarray',
                          'demo.levels/0.zarr/lat_bnds/.zattrs',
                          'demo.levels/0.zarr/lat_bnds/0.0',
                          'demo.levels/0.zarr/lon/.zarray'],
                         list(bucket.keys())[0:10])

        self.assertTrue('demo.levels/0.zarr/lat/.zarray' in bucket)
        self.assertTrue('demo.levels/0.zarr/lat_bnds/0.0' in bucket)
        self.assertTrue('demi.levels/0.zarr/lat_bnds/0.0' not in bucket)
