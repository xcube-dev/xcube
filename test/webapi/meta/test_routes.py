# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class MetaRoutesTest(RoutesTestCase):
    def test_fetch_server_info(self):
        response = self.fetch("/")
        self.assertResponseOK(response)

    def test_fetch_openapi_json(self):
        response = self.fetch("/openapi.json")
        self.assertResponseOK(response)

    def test_fetch_openapi_html(self):
        response = self.fetch("/openapi.html")
        self.assertResponseOK(response)

    def test_fetch_maintenance_fail(self):
        response = self.fetch("/maintenance/fail")
        self.assertResponse(
            response,
            expected_status=500,
            expected_message="Error! No worries, this is just a test.",
        )

        response = self.fetch("/maintenance/fail?message=HELP")
        self.assertResponse(response, expected_status=500, expected_message="HELP")

        response = self.fetch("/maintenance/fail?code=488")
        self.assertResponse(
            response,
            expected_status=488,
            expected_message="Error! No worries, this is just a test.",
        )

        response = self.fetch("/maintenance/fail?code=508&message=HELP")
        self.assertResponse(response, expected_status=508, expected_message="HELP")

        response = self.fetch("/maintenance/fail?code=x")
        self.assertResponse(
            response,
            expected_status=400,
            expected_message="HTTP 400: Bad Request"
            " (Query parameter 'code' must have type 'int'.)",
        )

    def test_fetch_maintenance_update(self):
        response = self.fetch("/maintenance/update")
        self.assertResponseOK(response)
