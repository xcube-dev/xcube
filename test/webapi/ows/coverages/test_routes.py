# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ...helpers import RoutesTestCase

from xcube.webapi.ows.coverages.routes import PATH_PREFIX


class CoveragesRoutesTest(RoutesTestCase):
    def test_fetch_coverage_json(self):
        response = self.fetch(
            PATH_PREFIX + "/collections/demo/coverage?f=application/json"
        )
        self.assertResponseOK(response)

    def test_fetch_coverage_html(self):
        response = self.fetch(
            PATH_PREFIX + "/collections/demo/coverage",
            headers=dict(
                Accept="text/nonexistent,application/json;q=0.9,text/html;q=1.0"
            ),
        )
        self.assertResponseOK(response)
        self.assertEqual("text/html", response.headers["Content-Type"])

    def test_fetch_coverage_netcdf(self):
        response = self.fetch(
            PATH_PREFIX + "/collections/demo/coverage?f=application/netcdf"
        )
        self.assertResponseOK(response)
        self.assertEqual(
            "50.00125,0.00125,52.49875,4.99875", response.headers["Content-Bbox"]
        )
        self.assertEqual("[EPSG:4326]", response.headers["Content-Crs"])

    def test_fetch_coverage_wrong_media_type(self):
        response = self.fetch(
            PATH_PREFIX + "/collections/demo/coverage",
            headers=dict(Accept="text/nonexistent"),
        )
        self.assertEqual(response.status, 415)

    def test_fetch_domainset(self):
        response = self.fetch(PATH_PREFIX + "/collections/demo/coverage/domainset")
        self.assertResponseOK(response)

    def test_fetch_rangetype(self):
        response = self.fetch(PATH_PREFIX + "/collections/demo/coverage/rangetype")
        self.assertResponseOK(response)

    def test_fetch_metadata(self):
        response = self.fetch(PATH_PREFIX + "/collections/demo/coverage/metadata")
        self.assertResponseOK(response)

    def test_fetch_rangeset(self):
        response = self.fetch(PATH_PREFIX + "/collections/demo/coverage/rangeset")
        self.assertEqual(response.status, 405)
