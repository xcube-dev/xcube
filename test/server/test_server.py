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
from typing import Optional, Sequence, Tuple, Callable, Dict, Any

from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.server.api import Api
from xcube.server.api import ApiContext
from xcube.server.api import ApiRoute
from xcube.server.context import Context
from xcube.server.framework import ServerFramework
from xcube.server.server import Server
from xcube.server.server import ServerContext
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema, JsonArraySchema


class ServerTest(unittest.TestCase):

    def test_web_server_delegation(self):
        extension_registry = new_extension_registry([
            ("datasets", dict()),
        ])
        web_server = MockServerFramework()
        server = Server(
            web_server, {},
            extension_registry=extension_registry
        )
        self.assertTrue(web_server.add_routes_called)
        self.assertTrue(web_server.update_called)
        self.assertFalse(web_server.start_called)
        self.assertFalse(web_server.stop_called)
        server.start()
        self.assertTrue(web_server.start_called)
        self.assertFalse(web_server.stop_called)
        server.stop()
        self.assertTrue(web_server.start_called)
        self.assertTrue(web_server.stop_called)

    def test_root_ctx(self):
        class SomeApiContext(ApiContext):
            def update(self, prev_ctx: Optional[ApiContext]):
                pass

        extension_registry = new_extension_registry([
            ("datasets", dict(create_ctx=SomeApiContext)),
        ])
        server = Server(
            MockServerFramework(), {},
            extension_registry=extension_registry
        )
        self.assertIsInstance(server.ctx, ServerContext)
        self.assertIsInstance(server.ctx.get_api_ctx('datasets'),
                              SomeApiContext)
        self.assertIsNone(server.ctx.get_api_ctx('timeseries'))
        self.assertEqual({'address': '0.0.0.0',
                          'port': 8080},
                         server.ctx.config)

    def test_config_schema(self):
        extension_registry = new_extension_registry([
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

    def test_update(self):
        extension_registry = new_extension_registry([
            ("datasets", dict()),
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

    def test_apis_loaded_in_order(self):
        extension_registry = new_extension_registry([
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
        self.assertIsInstance(apis, dict)
        self.assertEqual(['datasets',
                          'places',
                          'timeseries',
                          'wcs',
                          'wmts',
                          'stac',
                          'openeo'],
                         list(apis.keys()))

    def test_illegal_api_context_detected(self):
        # noinspection PyUnusedLocal
        def create_ctx(root_ctx):
            return 42

        extension_registry = new_extension_registry(
            [('datasets', dict())],
            create_ctx=create_ctx
        )

        with self.assertRaises(TypeError) as cm:
            Server(MockServerFramework(),
                   {},
                   extension_registry=extension_registry)
        self.assertEqual("API 'datasets':"
                         " context must be instance of ApiContext",
                         f'{cm.exception}')

    def test_missing_dependency_detected(self):
        extension_registry = new_extension_registry(
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
        self.add_routes_called = False
        self.update_called = False
        self.start_called = False
        self.stop_called = False

    def add_routes(self, routes: Sequence[ApiRoute]):
        self.add_routes_called = True

    def update(self, ctx: Context):
        self.update_called = True

    def start(self, ctx: Context):
        self.start_called = True

    def stop(self, ctx: Context):
        self.stop_called = True


def new_extension_registry(
        api_spec: Sequence[Tuple[str, Dict[str, Any]]],
        config_schema: Optional[JsonObjectSchema] = None,
        create_ctx: Optional[Callable] = None,
) -> ExtensionRegistry:
    extension_registry = ExtensionRegistry()

    # if config_schema is None:
    #     config_schema = JsonObjectSchema(additional_properties=True,
    #                                      default={})
    #
    # if create_ctx is None:
    #     class SomeApiContext(ApiContext):
    #
    #         def update(self, prev_ctx: Optional[ApiContext]):
    #             pass
    #
    #     create_ctx = SomeApiContext

    for api_name, api_kwargs in api_spec:
        api_kwargs = dict(api_kwargs)
        if config_schema is not None:
            api_kwargs.update(config_schema=config_schema)
        if create_ctx is not None:
            api_kwargs.update(create_ctx=create_ctx)
        api = Api(api_name, **api_kwargs)
        extension_registry.add_extension(EXTENSION_POINT_SERVER_APIS,
                                         api.name,
                                         component=api)

    return extension_registry
