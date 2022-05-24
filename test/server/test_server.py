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

from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.server.api import Api, ApiContext
from xcube.server.server import Server
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema


class ServerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.io_loop = self.new_io_loop()
        self.extension_registry = self.new_extension_registry()
        # noinspection PyTypeChecker
        self.server = Server({},
                             io_loop=self.io_loop,
                             extension_registry=self.extension_registry)

    def test_start_and_stop(self):
        io_loop = self.io_loop
        self.assertFalse(io_loop.start_called)
        self.assertFalse(io_loop.stop_called)
        self.server.start()
        self.assertTrue(io_loop.start_called)
        self.assertFalse(io_loop.stop_called)
        self.server.stop()
        self.assertTrue(io_loop.start_called)
        self.assertTrue(io_loop.stop_called)

    def test_ctx(self):
        server = self.server
        self.assertEqual({'address': '0.0.0.0',
                          'port': 8080},
                         server.ctx.config)

    def test_update(self):
        server = self.server
        prev_ctx = server.ctx
        server.update({"port": 9090})
        self.assertEqual({'address': '0.0.0.0',
                          'port': 9090},
                         server.ctx.config)
        self.assertIsNot(prev_ctx, server.ctx)

    def test_load_apis(self):
        apis = Server.load_apis(
            extension_registry=self.extension_registry
        )
        self.assertIsInstance(apis, dict)
        self.assertEqual(['datasets',
                          'places',
                          'timeseries',
                          'wcs', 'wmts',
                          'stac',
                          'openeo'],
                         list(apis.keys()))

    @staticmethod
    def new_io_loop():
        class IOLoopMock:
            def __init__(self):
                self.start_called = False
                self.stop_called = False

            def start(self):
                self.start_called = True

            def stop(self):
                self.stop_called = True

        return IOLoopMock()

    @staticmethod
    def new_extension_registry():
        extension_registry = ExtensionRegistry()

        config_schema = JsonObjectSchema(additional_properties=True,
                                         default={})

        for api_name, api_deps in (
                ("datasets", ()),
                ("places", ()),
                ("timeseries", ()),
                ("stac", ("datasets", "places")),
                ("openeo", ("datasets", "places", "timeseries")),
                ("wcs", ("datasets",)),
                ("wmts", ("datasets",)),
        ):
            class SomeApiContext(ApiContext):

                def update(self, prev_ctx: Optional[ApiContext]):
                    pass

            api = Api(api_name,
                      required_apis=api_deps,
                      config_schema=config_schema,
                      api_ctx_cls=SomeApiContext)
            extension_registry.add_extension(EXTENSION_POINT_SERVER_APIS,
                                             api.name,
                                             component=api)

        return extension_registry
