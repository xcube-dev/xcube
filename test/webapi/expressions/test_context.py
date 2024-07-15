# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Union

from test.webapi.helpers import get_api_ctx
from xcube.server.api import Context
from xcube.server.api import ServerConfig
from xcube.webapi.expressions.context import ExpressionsContext


def get_expressions_ctx(
    server_config: Union[str, ServerConfig] = "config.yml"
) -> ExpressionsContext:
    return get_api_ctx("expressions", ExpressionsContext, server_config)


class MetaContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_expressions_ctx()
        self.assertIsInstance(ctx, ExpressionsContext)
        self.assertIsInstance(ctx.datasets_ctx, Context)
