# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Union

from test.webapi.helpers import get_api_ctx
from xcube.server.api import ServerConfig
from xcube.webapi.statistics.context import StatisticsContext
from xcube.webapi.statistics.controllers import compute_statistics


def get_tiles_ctx(
    server_config: Union[str, ServerConfig] = "config.yml"
) -> StatisticsContext:
    return get_api_ctx("statistics", StatisticsContext, server_config)


class StatisticsControllerTest(unittest.TestCase):
    def test_compute_statistics(self):
        ctx = get_tiles_ctx()
        result = compute_statistics(ctx, "demo", "conc_tsm", {})
        self.assertIsInstance(result, dict)
        self.assertEqual({}, result)
