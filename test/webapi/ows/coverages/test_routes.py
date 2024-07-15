# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ...helpers import RoutesTestCase

from xcube.webapi.ows.coverages.routes import PATH_PREFIX
_COVERAGE_PREFIX = PATH_PREFIX + "/collections/demo/coverage"


class CoveragesRoutesTest(RoutesTestCase):
    def test_fetch_coverage_json(self):
        response = self.fetch(
            _COVERAGE_PREFIX + "?f=application/json"
        )
        self.assertResponseOK(response)

    def test_fetch_coverage_html(self):
        response = self.fetch(
            _COVERAGE_PREFIX + "",
            headers=dict(
                Accept="text/nonexistent,application/json;q=0.9,text/html;q=1.0"
            ),
        )
        self.assertResponseOK(response)
        self.assertEqual("text/html", response.headers["Content-Type"])

    def test_fetch_coverage_netcdf(self):
        response = self.fetch(
            _COVERAGE_PREFIX + "?f=application/netcdf"
        )
        self.assertResponseOK(response)
        self.assertEqual(
            "50.00125,0.00125,52.49875,4.99875", response.headers["Content-Bbox"]
        )
        self.assertEqual("[EPSG:4326]", response.headers["Content-Crs"])

    def test_fetch_coverage_wrong_media_type(self):
        response = self.fetch(
            _COVERAGE_PREFIX,
            headers=dict(Accept="text/nonexistent"),
        )
        self.assertEqual(response.status, 415)

    def test_fetch_domainset(self):
        response = self.fetch(_COVERAGE_PREFIX + "/domainset")
        self.assertResponseOK(response)

    def test_fetch_rangetype(self):
        response = self.fetch(_COVERAGE_PREFIX + "/rangetype")
        self.assertResponseOK(response)

    def test_fetch_metadata(self):
        response = self.fetch(_COVERAGE_PREFIX + "/metadata")
        self.assertResponseOK(response)

    def test_fetch_rangeset(self):
        response = self.fetch(_COVERAGE_PREFIX + "/rangeset")
        self.assertEqual(response.status, 405)
