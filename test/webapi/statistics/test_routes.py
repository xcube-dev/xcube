# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class StatisticsRoutesTest(RoutesTestCase):
    def test_fetch_statistics(self):
        response = self.fetch("/statistics/demo/conc_chl", method="POST")
        self.assertResponseOK(response)
