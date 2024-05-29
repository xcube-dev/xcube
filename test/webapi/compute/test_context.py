# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Union, Any
from collections.abc import Mapping

from test.webapi.helpers import get_api_ctx
from xcube.server.api import Context
from xcube.webapi.compute.context import ComputeContext, is_job_status


def get_compute_ctx(
    server_config: Union[str, Mapping[str, Any]] = "config.yml"
) -> ComputeContext:
    return get_api_ctx("compute", ComputeContext, server_config)


class ComputeContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_compute_ctx()
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.datasets_ctx, Context)
        self.assertIsInstance(ctx.places_ctx, Context)

    def test_is_status_with_valid_status(self):
        self.assertTrue(is_job_status(dict(state=dict(status="failed")), "failed"))

    def test_is_status_with_invalid_status(self):
        with self.assertRaises(ValueError) as test_context:
            is_job_status(dict(state=dict(status="failed")), "not_a_valid_status")
