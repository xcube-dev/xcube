# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class ExpressionsRoutesTest(RoutesTestCase):
    def test_fetch_expressions_capabilities(self):
        response = self.fetch("/expressions/capabilities")
        self.assertResponseOK(response)

    def test_fetch_expressions_evaluate_ok(self):
        response = self.fetch("/expressions/validate/demo/2*conc_chl")
        self.assertResponseOK(response)

    def test_fetch_expressions_evaluate_fail(self):
        response = self.fetch("/expressions/validate/demo/bibo*conc_chl")
        self.assertBadRequestResponse(response)

        result = response.json()
        print(result)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        error = result["error"]
        self.assertIsInstance(error, dict)
        self.assertEqual(400, error.get("status_code"))
        self.assertEqual(
            "HTTP 400: Bad Request (name 'bibo' is not defined)", error.get("message")
        )
        self.assertIn("exception", error)
        self.assertIsInstance(error["exception"], list)
