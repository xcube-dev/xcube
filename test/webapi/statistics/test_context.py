# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Union

from test.webapi.helpers import get_api_ctx
from xcube.server.api import Context
from xcube.server.api import ServerConfig
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.statistics.context import StatisticsContext


def get_statistics_ctx(
    server_config: Union[str, ServerConfig] = "config.yml"
) -> StatisticsContext:
    return get_api_ctx("statistics", StatisticsContext, server_config)


class StatisticsContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_statistics_ctx()
        self.assertIsInstance(ctx, StatisticsContext)
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.datasets_ctx, DatasetsContext)
