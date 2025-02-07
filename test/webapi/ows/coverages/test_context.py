# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


import unittest
from collections.abc import Mapping
from test.webapi.helpers import get_api_ctx
from typing import Any, Optional, Union

from xcube.server.api import Context
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.ows.coverages.context import CoveragesContext


def get_coverages_ctx(
    server_config: Optional[Union[str, Mapping[str, Any]]] = None,
) -> CoveragesContext:
    return get_api_ctx("ows.coverages", CoveragesContext, server_config)


class CoveragesContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_coverages_ctx()
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.datasets_ctx, DatasetsContext)
