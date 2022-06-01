# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import unittest
from typing import Optional

from tornado import concurrent

from xcube.server.server import Server
from xcube.server.server import ServerContext
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .mocks import MockApiContext
from .mocks import MockServerFramework
from .mocks import mock_extension_registry


class ServerTest(unittest.TestCase):

    def test_web_server_delegation(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict(create_ctx=MockApiContext)),
        ])
        framework = MockServerFramework()
        server = Server(
            framework, {},
            extension_registry=extension_registry
        )
        self.assertEqual(1, framework.add_routes_count)
        self.assertEqual(1, framework.update_count)
        self.assertEqual(0, framework.start_count)
        self.assertEqual(0, framework.stop_count)
        server.start()
        self.assertEqual(1, framework.start_count)
        self.assertEqual(0, framework.stop_count)
        server.stop()
        self.assertEqual(1, framework.start_count)
        self.assertEqual(1, framework.stop_count)

    def test_root_ctx(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict(create_ctx=MockApiContext)),
        ])
        server = Server(
            MockServerFramework(), {},
            extension_registry=extension_registry
        )
        self.assertIsInstance(server.server_ctx, ServerContext)
        self.assertIsInstance(server.server_ctx.get_api_ctx('datasets'),
                              MockApiContext)
        self.assertTrue(server.server_ctx.get_api_ctx('datasets').on_update_count)
        self.assertIsNone(server.server_ctx.get_api_ctx('timeseries'))
        self.assertEqual({'address': '0.0.0.0',
                          'port': 8080},
                         server.server_ctx.config)

    def test_config_schema_effectively_merged(self):
        extension_registry = mock_extension_registry([
            (
                "datasets",
                dict(
                    config_schema=JsonObjectSchema(
                        properties=dict(
                            data_stores=JsonArraySchema(
                                items=JsonObjectSchema(
                                    additional_properties=True
                                )
                            )
                        ),
                        required=['data_stores'],
                        additional_properties=False
                    ))
            ),
        ])
        server = Server(
            MockServerFramework(),
            {
                "data_stores": []
            },
            extension_registry=extension_registry
        )
        self.assertIsInstance(server.config_schema, JsonObjectSchema)
        self.assertEqual(
            {
                'type': 'object',
                'properties': {
                    'address': {
                        'type': 'string',
                        'default': '0.0.0.0'
                    },
                    'port': {
                        'type': 'integer',
                        'default': 8080
                    },
                    'data_stores': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'additionalProperties': True,
                        }
                    },
                },
                'required': ['data_stores'],
                'additionalProperties': True,
            },
            server.config_schema.to_dict()
        )

    def test_config_schema_must_be_object(self):
        extension_registry = mock_extension_registry([
            (
                "datasets",
                dict(
                    config_schema=JsonObjectSchema(
                        properties=dict(address=JsonStringSchema())
                    )
                )
            ),
        ])
        with self.assertRaises(ValueError) as cm:
            Server(
                MockServerFramework(), {},
                extension_registry=extension_registry
            )
        self.assertEqual(f"API 'datasets':"
                         f" configuration parameter 'address'"
                         f" is already defined.",
                         f'{cm.exception}')

    def test_update_is_effective(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict(create_ctx=MockApiContext)),
            ("timeseries", dict(create_ctx=MockApiContext,
                                required_apis=["datasets"])),
        ])
        server = Server(
            MockServerFramework(), {},
            extension_registry=extension_registry
        )
        prev_ctx = server.server_ctx
        server.update({"port": 9090})
        self.assertEqual({'address': '0.0.0.0',
                          'port': 9090},
                         server.server_ctx.config)
        self.assertIsNot(prev_ctx, server.server_ctx)

    def test_update_disposes(self):
        api_ctx: Optional[MockApiContext] = None

        def create_ctx(root):
            nonlocal api_ctx
            if api_ctx is None:
                api_ctx = MockApiContext(root)
                return api_ctx
            else:
                return None

        extension_registry = mock_extension_registry([
            ("datasets", dict(create_ctx=create_ctx)),
        ])
        server = Server(
            MockServerFramework(), {},
            extension_registry=extension_registry
        )
        self.assertIsInstance(api_ctx, MockApiContext)
        self.assertEqual(1, api_ctx.on_update_count)
        self.assertEqual(0, api_ctx.on_dispose_count)
        server.update({})
        self.assertEqual(1, api_ctx.on_update_count)
        self.assertEqual(1, api_ctx.on_dispose_count)

    def test_call_later(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict()),
        ])
        framework = MockServerFramework()
        server = Server(
            framework, {},
            extension_registry=extension_registry
        )
        self.assertEqual(0, framework.call_later_count)
        result = server.call_later(0.01, lambda x: x)
        self.assertIsInstance(result, object)
        self.assertEqual(1, framework.call_later_count)

    def test_run_in_executor(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict()),
        ])
        framework = MockServerFramework()
        server = Server(
            framework, {},
            extension_registry=extension_registry
        )
        self.assertEqual(0, framework.run_in_executor_count)
        result = server.run_in_executor(None, lambda x: x)
        self.assertIsInstance(result, concurrent.futures.Future)
        self.assertEqual(1, framework.run_in_executor_count)

    def test_apis_loaded_in_order(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict()),
            ("places", dict()),
            ("timeseries", dict()),
            ("stac", dict(required_apis=("datasets", "places"))),
            ("openeo", dict(required_apis=("datasets",
                                           "places", "timeseries"))),
            ("wcs", dict(required_apis=("datasets",))),
            ("wmts", dict(required_apis=("datasets",))),
        ])
        apis = Server.load_apis(extension_registry=extension_registry)
        self.assertIsInstance(apis, tuple)
        self.assertEqual(('datasets',
                          'places',
                          'timeseries',
                          'wcs',
                          'wmts',
                          'stac',
                          'openeo'),
                         tuple(api.name for api in apis))

    def test_illegal_api_context_detected(self):
        # noinspection PyUnusedLocal
        def create_ctx(root_ctx):
            return 42

        extension_registry = mock_extension_registry(
            [('datasets', dict(create_ctx=create_ctx))],
        )

        with self.assertRaises(TypeError) as cm:
            Server(MockServerFramework(),
                   {},
                   extension_registry=extension_registry)
        self.assertEqual("API 'datasets':"
                         " context must be instance of ApiContext",
                         f'{cm.exception}')

    def test_missing_dependency_detected(self):
        extension_registry = mock_extension_registry(
            [('timeseries', dict(required_apis=('datasets',)))]
        )

        with self.assertRaises(ValueError) as cm:
            Server(MockServerFramework(),
                   {},
                   extension_registry=extension_registry)
        self.assertEqual("API 'timeseries':"
                         " missing API dependency 'datasets'",
                         f'{cm.exception}')

