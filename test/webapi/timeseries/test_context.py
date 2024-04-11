# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Union

from test.webapi.helpers import get_api_ctx
from xcube.server.api import Context
from xcube.server.api import ServerConfig
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.timeseries.context import TimeSeriesContext


def get_timeseries_ctx(
    server_config: Union[str, ServerConfig] = "config.yml"
) -> TimeSeriesContext:
    return get_api_ctx("timeseries", TimeSeriesContext, server_config)


class DatasetsContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_timeseries_ctx()
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.datasets_ctx, DatasetsContext)
