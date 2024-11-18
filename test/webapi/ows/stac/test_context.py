# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


import unittest
from typing import Optional, Union, Any
from collections.abc import Mapping

from test.webapi.helpers import get_api_ctx
from xcube.server.api import Context
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.ows.stac.context import StacContext


def get_stac_ctx(
    server_config: Optional[Union[str, Mapping[str, Any]]] = None
) -> StacContext:
    return get_api_ctx("ows.stac", StacContext, server_config)


class StacContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_stac_ctx()
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.datasets_ctx, DatasetsContext)
