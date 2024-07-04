# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class ExpressionsRoutesTest(RoutesTestCase):

    def test_fetch_expressions_capabilities(self):
        response = self.fetch("/expressions/capabilities")
        self.assertResponseOK(response)

    def test_fetch_expressions_evaluate_ok(self):
        response = self.fetch("/expressions/evaluate/demo/2*conc_chl")
        self.assertResponseOK(response)

    def test_fetch_expressions_evaluate_fail(self):
        response = self.fetch("/expressions/evaluate/demo/bibo*conc_chl")
        self.assertBadRequestResponse(response)

        error = response.json()
        self.assertIsInstance(error, dict)
        self.assertIn("error", error)
        self.assertIn("exception", error["error"])
        self.assertIn(
            "xcube.core.varexpr.VarExprError: name 'bibo' is not defined\n",
            error["error"]["exception"],
        )
