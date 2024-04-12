# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Union

from test.webapi.helpers import get_api_ctx
from xcube.server.api import Context
from xcube.server.api import ServerConfig
from xcube.webapi.meta.context import MetaContext


def get_meta_ctx(server_config: Union[str, ServerConfig] = "config.yml") -> MetaContext:
    return get_api_ctx("meta", MetaContext, server_config)


class MetaContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_meta_ctx()
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.start_time, str)
        self.assertIsInstance(ctx.update_time, str)
        self.assertIsInstance(ctx.current_time, str)
        self.assertGreaterEqual(ctx.update_time, ctx.start_time)
        self.assertGreaterEqual(ctx.current_time, ctx.update_time)
