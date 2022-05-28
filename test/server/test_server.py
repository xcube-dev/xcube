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
from typing import Optional, Sequence, Tuple, Dict, Any, Union, Callable

from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.server.api import Api
from xcube.server.api import ApiContext
from xcube.server.api import ApiRoute
from xcube.server.context import Context
from xcube.server.framework import ServerFramework
from xcube.server.server import Server
from xcube.server.server import ServerContext
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema, JsonArraySchema, \
    JsonStringSchema


class ServerTest(unittest.TestCase):

    def test_web_server_delegation(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict(create_ctx=MockApiContext)),
        ])
        web_server = MockServerFramework()
        server = Server(
            web_server, {},
            extension_registry=extension_registry
        )
        self.assertEqual(1, web_server.add_routes_count)
        self.assertEqual(1, web_server.update_count)
        self.assertEqual(0, web_server.start_count)
        self.assertEqual(0, web_server.stop_count)
        server.start()
        self.assertEqual(1, web_server.start_count)
        self.assertEqual(0, web_server.stop_count)
        server.stop()
        self.assertEqual(1, web_server.start_count)
        self.assertEqual(1, web_server.stop_count)

    def test_root_ctx(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict(create_ctx=MockApiContext)),
        ])
        server = Server(
            MockServerFramework(), {},
            extension_registry=extension_registry
        )
        self.assertIsInstance(server.ctx, ServerContext)
        self.assertIsInstance(server.ctx.get_api_ctx('datasets'),
                              MockApiContext)
        self.assertTrue(server.ctx.get_api_ctx('datasets').update_count)
        self.assertIsNone(server.ctx.get_api_ctx('timeseries'))
        self.assertEqual({'address': '0.0.0.0',
                          'port': 8080},
                         server.ctx.config)

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
                'additionalProperties': False,
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
        prev_ctx = server.ctx
        server.update({"port": 9090})
        self.assertEqual({'address': '0.0.0.0',
                          'port': 9090},
                         server.ctx.config)
        self.assertIsNot(prev_ctx, server.ctx)

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
        self.assertEqual(1, api_ctx.update_count)
        self.assertEqual(0, api_ctx.dispose_count)
        server.update({})
        self.assertEqual(1, api_ctx.update_count)
        self.assertEqual(1, api_ctx.dispose_count)

    def test_call_later(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict()),
        ])
        framework = MockServerFramework()
        server = Server(
            framework, {},
            extension_registry=extension_registry
        )
        server.call_later(0.01, lambda x: x)
        self.assertEqual(1, framework.call_later_count)

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
        self.assertIsInstance(apis, list)
        self.assertEqual(['datasets',
                          'places',
                          'timeseries',
                          'wcs',
                          'wmts',
                          'stac',
                          'openeo'],
                         [api.name for api in apis])

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


class MockServerFramework(ServerFramework):

    def __init__(self):
        self.add_routes_count = 0
        self.update_count = 0
        self.start_count = 0
        self.stop_count = 0
        self.call_later_count = 0

    def add_routes(self, routes: Sequence[ApiRoute]):
        self.add_routes_count += 1

    def update(self, ctx: Context):
        self.update_count += 1

    def start(self, ctx: Context):
        self.start_count += 1

    def stop(self, ctx: Context):
        self.stop_count += 1

    def call_later(self,
                   delay: Union[int, float],
                   callback: Callable,
                   *args,
                   **kwargs):
        self.call_later_count += 1


class MockApiContext(ApiContext):
    def __init__(self, root: Context):
        super().__init__(root)
        self.update_count = 0
        self.dispose_count = 0

    def update(self, prev_ctx: Optional[ApiContext]):
        self.update_count += 1

    def dispose(self):
        self.dispose_count += 1


def mock_extension_registry(
        api_spec: Sequence[Tuple[str, Dict[str, Any]]]
) -> ExtensionRegistry:
    extension_registry = ExtensionRegistry()
    for api_name, api_kwargs in api_spec:
        api_kwargs = dict(api_kwargs)
        api = Api(api_name, **api_kwargs)
        extension_registry.add_extension(EXTENSION_POINT_SERVER_APIS,
                                         api.name,
                                         component=api)
    return extension_registry
