# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class TilesRoutesTest(RoutesTestCase):
    def test_fetch_dataset_tile(self):
        response = self.fetch("/tiles/demo/conc_chl/0/0/0")
        self.assertResponseOK(response)

    def test_fetch_dataset_rgb_tile(self):
        response = self.fetch(
            "/tiles/demo/rgb/0/0/0?" "r=kd489&g=conc_chl&b=conc_tsm&time=first"
        )
        self.assertResponseOK(response)

    def test_fetch_dataset_tile_with_params(self):
        response = self.fetch(
            "/tiles/demo/conc_chl/0/0/0?" "time=current&cmap=jet&debug=1"
        )
        self.assertResponseOK(response)
