# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.webapi.statistics.controllers import compute_statistics
from .test_context import get_statistics_ctx


class StatisticsControllerTest(unittest.TestCase):
    def test_compute_statistics_for_point(self):
        lon = 1.768
        lat = 51.465
        time = "2017-01-16 10:09:21"

        ctx = get_statistics_ctx()
        dataset = ctx.datasets_ctx.get_dataset("demo")
        expected_value = float(
            dataset["conc_tsm"]
            .sel(lon=lon, lat=lat, time=time, method="nearest")
            .values
        )

        result = compute_statistics(
            ctx,
            "demo",
            "conc_tsm",
            {"type": "Point", "coordinates": [lon, lat]},
            {"time": time},
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(
            {
                "count": 1,
                "deviation": 0.0,
                "maximum": expected_value,
                "mean": expected_value,
                "minimum": expected_value,
            },
            result,
        )
