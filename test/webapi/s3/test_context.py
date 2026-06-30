# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from test.webapi.helpers import get_api_ctx
from typing import Union

from xcube.server.api import ServerConfig
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.s3.context import S3Context
from xcube.webapi.s3.objectstorage import ObjectStorage


def get_s3_ctx(server_config: Union[str, ServerConfig] = "config.yml") -> S3Context:
    return get_api_ctx("s3", S3Context, server_config)


class S3ContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_s3_ctx()
        self.assertIsInstance(ctx.datasets_ctx, DatasetsContext)
        self.assertIsInstance(ctx.get_bucket("datasets"), ObjectStorage)
        self.assertIsInstance(ctx.get_bucket("pyramids"), ObjectStorage)

    def test_datasets_bucket(self):
        ctx = get_s3_ctx()
        bucket = ctx.get_bucket("datasets")
        self.assertCountEqual(
            [
                "demo.zarr/c2rcc_flags",
                "demo.zarr/conc_chl",
                "demo.zarr/conc_tsm",
                "demo.zarr/kd489",
                "demo.zarr/quality_flags",
                "demo-1w.zarr/c2rcc_flags",
                "demo-1w.zarr/conc_chl",
                "demo-1w.zarr/conc_tsm",
                "demo-1w.zarr/kd489",
                "demo-1w.zarr/quality_flags",
                "demo-1w.zarr/c2rcc_flags_stdev",
                "demo-1w.zarr/conc_chl_stdev",
                "demo-1w.zarr/conc_tsm_stdev",
                "demo-1w.zarr/kd489_stdev",
                "demo-1w.zarr/quality_flags_stdev",
                "demo-multidimensional.zarr/conc_chl",
            ],
            list(bucket.keys()),
        )

    def test_pyramids_bucket(self):
        ctx = get_s3_ctx()
        bucket = ctx.get_bucket("pyramids")
        self.assertEqual(
            [
                "demo.levels/0.zarr/c2rcc_flags",
                "demo.levels/0.zarr/conc_chl",
                "demo.levels/0.zarr/conc_tsm",
                "demo.levels/0.zarr/kd489",
                "demo.levels/0.zarr/quality_flags",
                "demo.levels/1.zarr/c2rcc_flags",
                "demo.levels/1.zarr/conc_chl",
                "demo.levels/1.zarr/conc_tsm",
                "demo.levels/1.zarr/kd489",
                "demo.levels/1.zarr/quality_flags",
            ],
            list(bucket.keys())[0:10],
        )

        self.assertTrue("demo-1w.levels/2.zarr/conc_tsm_stdev" in bucket)
        self.assertTrue("demo-multidimensional.levels/2.zarr/conc_chl" in bucket)
        self.assertFalse("demo-multidimensional.levels/3.zarr/conc_chl" in bucket)
