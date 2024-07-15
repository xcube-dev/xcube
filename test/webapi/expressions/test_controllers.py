# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Union

from test.mixins import AlmostEqualDeepMixin
from test.webapi.helpers import get_api_ctx
from xcube.server.api import ServerConfig
from xcube.webapi.expressions.context import ExpressionsContext
from xcube.webapi.expressions.controllers import get_expressions_capabilities


def get_expressions_ctx(
    server_config: Union[str, ServerConfig] = "config.yml"
) -> ExpressionsContext:
    return get_api_ctx("expressions", ExpressionsContext, server_config)


class ExpressionsControllerTest(unittest.TestCase, AlmostEqualDeepMixin):

    def test_get_expressions_capabilities(self):
        ctx = get_expressions_ctx()
        result = get_expressions_capabilities(ctx)
        self.assertIsInstance(result, dict)
        self.assertIn("namespace", result)
        self.assertIsInstance(result["namespace"], dict)
        self.assertEqual(
            {
                "arrayFunctions",
                "otherFunctions",
                "arrayOperators",
                "otherOperators",
                "constants",
            },
            set(result["namespace"].keys()),
        )
