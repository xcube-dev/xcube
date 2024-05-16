# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.webapi.statistics.controllers import compute_statistics
from .test_context import get_statistics_ctx


class StatisticsControllerTest(unittest.TestCase):
    def test_compute_statistics(self):
        ctx = get_statistics_ctx()
        result = compute_statistics(ctx, "demo", "conc_tsm", {})
        self.assertIsInstance(result, dict)
        self.assertEqual({}, result)
