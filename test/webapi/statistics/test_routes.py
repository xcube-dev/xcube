# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class StatisticsRoutesTest(RoutesTestCase):
    def test_fetch_statistics_ok(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?time=2017-01-16+10:09:21",
            method="POST",
            body='{"type": "Point", "coordinates": [1.768, 51.465]}',
        )
        self.assertResponseOK(response)

    def test_fetch_statistics_missing_time(self):
        response = self.fetch(
            "/statistics/demo/conc_chl",
            method="POST",
            body='{"type": "Point", "coordinates": [1.768, 51.465]}',
        )
        self.assertBadRequestResponse(response, "Missing query parameter 'time'")
