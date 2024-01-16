# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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
from typing import Union, Mapping, Any

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
        self.assertTrue(
            is_job_status(
                dict(state=dict(status='failed')),
                'failed'
            )
        )

    def test_is_status_with_invalid_status(self):
        with self.assertRaises(ValueError) as test_context:
            is_job_status(
                dict(state=dict(status='failed')),
                'not_a_valid_status'
            )
