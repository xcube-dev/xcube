# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class DatasetsRoutesTest(RoutesTestCase):
    def test_fetch_datasets(self):
        response = self.fetch("/datasets")
        self.assertResponseOK(response)

    def test_fetch_datasets_details(self):
        response = self.fetch("/datasets?details=1")
        self.assertResponseOK(response)
        response = self.fetch("/datasets?details=1&tiles=ol")
        self.assertResponseOK(response)
        response = self.fetch("/datasets?details=1&point=2.8,51.0")
        self.assertResponseOK(response)
        response = self.fetch("/datasets?details=1&point=2,8a,51.0")
        self.assertBadRequestResponse(
            response, "illegal point: could not convert string to float: '8a'"
        )

    def test_fetch_dataset(self):
        response = self.fetch("/datasets/demo")
        self.assertResponseOK(response)
        response = self.fetch("/datasets/demo?tiles=ol4")
        self.assertResponseOK(response)

    def test_fetch_dataset_coords(self):
        response = self.fetch("/datasets/demo/coords/time")
        self.assertResponseOK(response)

    def test_fetch_legend(self):
        # New
        response = self.fetch("/tiles/demo/conc_chl/legend")
        self.assertResponseOK(response)
        # Old
        response = self.fetch("/datasets/demo/vars/conc_chl/legend.png")
        self.assertResponseOK(response)
