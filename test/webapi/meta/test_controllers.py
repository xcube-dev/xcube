# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from test.mixins import AlmostEqualDeepMixin
from test.webapi.helpers import get_api_ctx
from xcube.webapi.meta.context import MetaContext
from xcube.webapi.meta.controllers import get_expressions_namespace
from xcube.webapi.meta.controllers import get_service_info


def get_meta_ctx(server_config=None) -> MetaContext:
    return get_api_ctx("meta", MetaContext, server_config)


class MetaControllerTest(unittest.TestCase, AlmostEqualDeepMixin):
    def test_get_service_info(self):
        ctx = get_meta_ctx()
        result = get_service_info(ctx)
        self.assertIsInstance(result, dict)
        self.assertEqual(
            {
                "apis",
                "currentTime",
                "description",
                "name",
                "pid",
                "startTime",
                "updateTime",
                "version",
                "versions",
            },
            set(result.keys()),
        )

    def test_get_expressions_namespace(self):
        ctx = get_meta_ctx()
        result = get_expressions_namespace()
        self.assertIsInstance(result, dict)
        self.assertIn("namespace", result)
        self.assertIsInstance(result["namespace"], dict)
        self.assertEqual(
            {
                "arrayFunctions",
                "arrayOperators",
                "builtinFunctions",
                "builtinOperators",
                "constants",
            },
            set(result["namespace"].keys()),
        )
