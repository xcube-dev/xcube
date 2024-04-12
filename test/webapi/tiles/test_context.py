# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Union

from test.webapi.helpers import get_api_ctx
from xcube.server.api import Context
from xcube.server.api import ServerConfig
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.tiles.context import TilesContext


def get_tiles_ctx(
    server_config: Union[str, ServerConfig] = "config.yml"
) -> TilesContext:
    return get_api_ctx("tiles", TilesContext, server_config)


class TilesContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_tiles_ctx()
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.datasets_ctx, DatasetsContext)
