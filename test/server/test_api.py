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

from xcube.server.api import Api
from xcube.server.api import ApiContext
from xcube.server.api import ApiHandler
from xcube.server.api import ApiRoute
from xcube.server.api import Context
from xcube.server.server import ServerContext
from .mocks import MockApiError
from .mocks import MockApiRequest
from .mocks import MockApiResponse
from .mocks import mock_server


class ApiTest(unittest.TestCase):

    def test_default_props(self):
        api = Api("datasets")
        self.assertEqual("datasets", api.name)
        self.assertEqual("0.0.0", api.version)
        self.assertEqual(None, api.description)
        self.assertEqual((), api.required_apis)
        self.assertEqual((), api.optional_apis)
        self.assertEqual(None, api.config_schema)
        self.assertEqual((), api.routes)

    def test_ctor_functions(self):
        class MyApiContext(ApiContext):
            def on_update(self, prev_ctx: Optional["Context"]):
                pass

        test_dict = dict()

        def handle_start(root):
            test_dict['handle_start'] = root

        def handle_stop(root):
            test_dict['handle_stop'] = root

        root_ctx = ServerContext(mock_server(), {})

        api = Api("datasets",
                  create_ctx=MyApiContext,
                  on_start=handle_start,
                  on_stop=handle_stop)

        api_ctx = api.create_ctx(root_ctx)
        self.assertIsInstance(api_ctx, MyApiContext)

        api.on_start(root_ctx)
        self.assertIs(root_ctx, test_dict.get('handle_start'))
        self.assertIs(None, test_dict.get('handle_stop'))

        api.on_stop(root_ctx)
        self.assertIs(root_ctx, test_dict.get('handle_start'))
        self.assertIs(root_ctx, test_dict.get('handle_stop'))

    def test_route_decorator(self):
        api = Api("datasets")

        @api.route("/datasets")
        class DatasetsHandler(ApiHandler):
            # noinspection PyMethodMayBeStatic
            def get(self):
                return {}

        @api.route("/datasets/{dataset_id}")
        class DatasetHandler(ApiHandler):
            # noinspection PyMethodMayBeStatic
            def get(self):
                return {}

        self.assertEqual(
            (
                ApiRoute("datasets", "/datasets", DatasetsHandler),
                ApiRoute("datasets", "/datasets/{dataset_id}", DatasetHandler)
            ),
            api.routes
        )

    def test_operation_decorator(self):
        api = Api("datasets")

        @api.route("/datasets")
        class DatasetsHandler(ApiHandler):
            # noinspection PyMethodMayBeStatic
            @api.operation()
            def get(self):
                return {}

        self.assertEqual(
            {},
            getattr(DatasetsHandler.get, '__openapi__', None)
        )


class ApiRouteTest(unittest.TestCase):

    def test_equal(self):
        class DatasetHandler(ApiHandler):
            # noinspection PyMethodMayBeStatic
            def get(self):
                return {}

        self.assertTrue(
            ApiRoute("datasets", "/datasets", DatasetHandler) ==
            ApiRoute("datasets", "/datasets", DatasetHandler)
        )
        self.assertTrue(
            ApiRoute("datasets", "/datasets", DatasetHandler) ==
            ApiRoute("datasets", "/datasets", DatasetHandler, {})
        )
        self.assertTrue(
            ApiRoute("datasets", "/datasets", DatasetHandler) !=
            ApiRoute("datasets", "/dataset", DatasetHandler, {})
        )
        self.assertTrue(
            ApiRoute("datasets", "/datasets", DatasetHandler) !=
            ApiRoute("dataset", "/datasets", DatasetHandler, {})
        )
        self.assertTrue(
            ApiRoute("datasets", "/datasets", DatasetHandler) != 42
        )

    def test_hash(self):
        class DatasetHandler(ApiHandler):
            # noinspection PyMethodMayBeStatic
            def get(self):
                return {}

        self.assertEqual(
            hash(ApiRoute("datasets", "/datasets", DatasetHandler)),
            hash(ApiRoute("datasets", "/datasets", DatasetHandler))
        )
        self.assertEqual(
            hash(ApiRoute("datasets", "/datasets", DatasetHandler)),
            hash(ApiRoute("datasets", "/datasets", DatasetHandler, {}))
        )

    def test_str_and_repr(self):
        class DatasetHandler(ApiHandler):
            # noinspection PyMethodMayBeStatic
            def get(self):
                return {}

        api_route = ApiRoute("datasets", "/datasets",
                             DatasetHandler, dict(force=True))
        self.assertEqual("ApiRoute("
                         "'datasets',"
                         " '/datasets',"
                         " DatasetHandler,"
                         " handler_kwargs={'force': True}"
                         ")",
                         f'{api_route}')
        self.assertEqual("ApiRoute("
                         "'datasets',"
                         " '/datasets',"
                         " DatasetHandler,"
                         " handler_kwargs={'force': True}"
                         ")",
                         f'{api_route!r}')


class ApiContextTest(unittest.TestCase):
    class DatasetsContext(ApiContext):

        def on_update(self, prev_ctx: Optional[Context]):
            pass

    class TimeSeriesContext(ApiContext):
        def __init__(self, root: Context):
            super().__init__(root)
            self.dataset_ctx = root.get_api_ctx("datasets")

        def on_update(self, prev_ctx: Optional[Context]):
            pass

    def test_it(self):
        api1 = Api("datasets",
                   create_ctx=self.DatasetsContext)
        api2 = Api("timeseries",
                   create_ctx=self.TimeSeriesContext,
                   required_apis=["datasets"])
        config = {}
        root_ctx = ServerContext(mock_server([api1, api2]), config)
        root_ctx.on_update(None)
        api1_ctx = root_ctx.get_api_ctx('datasets')
        api2_ctx = root_ctx.get_api_ctx('timeseries')

        self.assertIsInstance(api1_ctx, self.DatasetsContext)
        self.assertIsInstance(api2_ctx, self.TimeSeriesContext)

        self.assertIs(root_ctx, api1_ctx.root)
        self.assertIs(root_ctx, api2_ctx.root)

        self.assertIs(config, api1_ctx.config)
        self.assertIs(config, api2_ctx.config)

        api21_ctx = api2_ctx.get_api_ctx("datasets")
        self.assertIsInstance(api21_ctx, self.DatasetsContext)


class ApiHandlerTest(unittest.TestCase):
    class DatasetsContext(ApiContext):

        def on_update(self, prev_ctx: Optional[Context]):
            pass

    def setUp(self) -> None:
        self.api = Api("datasets", create_ctx=self.DatasetsContext)
        self.config = {}
        self.root_ctx = ServerContext(mock_server([self.api]), self.config)
        self.root_ctx.on_update(None)
        self.request = MockApiRequest()
        self.response = MockApiResponse()
        self.handler = ApiHandler("datasets",
                                  self.root_ctx,
                                  self.request,
                                  self.response)

    def test_props(self):
        handler = self.handler
        self.assertIs(self.root_ctx, handler.root_ctx)
        self.assertIs(self.request, handler.request)
        self.assertIs(self.response, handler.response)
        self.assertIs(self.config, handler.config)
        self.assertIsInstance(handler.ctx, self.DatasetsContext)

    def test_default_methods(self):
        handler = self.handler

        with self.assertRaises(MockApiError) as cm:
            handler.get()
        self.assertEqual(405, cm.exception.args[0])

        with self.assertRaises(MockApiError) as cm:
            handler.post()
        self.assertEqual(405, cm.exception.args[0])

        with self.assertRaises(MockApiError) as cm:
            handler.put()
        self.assertEqual(405, cm.exception.args[0])

        with self.assertRaises(MockApiError) as cm:
            handler.delete()
        self.assertEqual(405, cm.exception.args[0])

        with self.assertRaises(MockApiError) as cm:
            handler.options()
        self.assertEqual(405, cm.exception.args[0])


class ApiRequestTest(unittest.TestCase):
    def test_query_args(self):
        request = MockApiRequest()
        self.assertEqual(['1'], request.get_query_args('details'))
        self.assertEqual('1', request.get_query_arg('details'))
        self.assertEqual([], request.get_query_args('crs'))
        self.assertEqual(None, request.get_query_arg('crs'))

    def test_body_args(self):
        request = MockApiRequest()
        self.assertEqual([bytes(10)], request.get_body_args('secret'))
        self.assertEqual(bytes(10), request.get_body_arg('secret'))
        self.assertEqual([], request.get_body_args('key'))
        self.assertEqual(None, request.get_body_arg('key'))

