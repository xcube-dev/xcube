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
import os
import unittest
from typing import Mapping, Any, Union
import collections.abc

from test.webapi.helpers import get_api_ctx
from xcube.webapi.common.context import ResourcesContext


def get_resource_ctx(
        server_config: Union[str, Mapping[str, Any]] = "config.yml"
) -> ResourcesContext:
    return get_api_ctx("datasets", ResourcesContext, server_config)


class ResourceContextTest(unittest.TestCase):

    def setUp(self) -> None:
        os.environ['USER_STORAGE_BUCKET'] = 'bibos_bucket'
        os.environ['USER_STORAGE_KEY'] = 'bibos_key'
        os.environ['USER_STORAGE_SECRET'] = 'bibos_secret'

    def tearDown(self) -> None:
        del os.environ['USER_STORAGE_BUCKET']
        del os.environ['USER_STORAGE_KEY']
        del os.environ['USER_STORAGE_SECRET']

    def test_env(self):
        ctx = get_resource_ctx()
        self.assertIsInstance(ctx.env, collections.abc.Mapping)
        self.assertIn('USER_STORAGE_BUCKET', ctx.env)

    def test_eval_config_value_with_env(self):
        ctx = get_resource_ctx()

        value = {
            "StoreParams": {
                "root": "${ctx.env['USER_STORAGE_BUCKET']}",
                "storage_params": {
                    "anon": False,
                    "key": "${ctx.env['USER_STORAGE_KEY']}",
                    "secret": "${ctx.env['USER_STORAGE_SECRET']}",
                }
            }
        }

        interpolated_value = {
            "StoreParams": {
                "root": "bibos_bucket",
                "storage_params": {
                    "anon": False,
                    "key": "bibos_key",
                    "secret": "bibos_secret",
                }
            }
        }

        self.assertEqual(interpolated_value,
                         ctx.eval_config_value(value))
