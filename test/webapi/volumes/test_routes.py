# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class VolumesRoutesTest(RoutesTestCase):
    def test_fetch_dataset_volume_ok(self):
        response = self.fetch("/volumes/demo/conc_chl?bbox=1.0,51.0,2.0,51.5")
        self.assertResponseOK(response)

    def test_fetch_dataset_volume_404(self):
        response = self.fetch("/volumes/demo/conc_x?bbox=1.0,51.0,2.0,51.5")
        self.assertResourceNotFoundResponse(response)

    def test_fetch_dataset_volume_400(self):
        response = self.fetch("/volumes/demo/conc_chl?bbox=1.0,51.0")
        self.assertBadRequestResponse(response)
