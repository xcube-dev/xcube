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
from xcube.server.api import ApiError
from xcube.server.server import ServerContext
from xcube.util.frozen import FrozenDict
from .mocks import MockApiRequest
from .mocks import MockApiResponse
from .mocks import MockFramework
from .mocks import mock_server


class ApiTest(unittest.TestCase):

    def test_basic_props(self):
        api = Api("datasets", description='What an API!')
        self.assertEqual("datasets", api.name)
        self.assertEqual("0.0.0", api.version)
        self.assertEqual('What an API!', api.description)
        self.assertEqual((), api.required_apis)
        self.assertEqual((), api.optional_apis)
        self.assertEqual(None, api.config_schema)
        self.assertEqual((), api.routes)

    def test_ctor_functions(self):
        class MyApiContext(ApiContext):
            def on_update(self, prev_ctx: Optional["Context"]):
                pass

        test_dict = dict()

        def handle_start(ctx):
            test_dict['handle_start'] = ctx

        def handle_stop(ctx):
            test_dict['handle_stop'] = ctx

        server_ctx = ServerContext(mock_server(), {})

        api = Api("datasets",
                  create_ctx=MyApiContext,
                  on_start=handle_start,
                  on_stop=handle_stop)

        api_ctx = api.create_ctx(server_ctx)
        self.assertIsInstance(api_ctx, MyApiContext)

        api.on_start(server_ctx)
        self.assertIs(server_ctx, test_dict.get('handle_start'))
        self.assertIs(None, test_dict.get('handle_stop'))

        api.on_stop(server_ctx)
        self.assertIs(server_ctx, test_dict.get('handle_start'))
        self.assertIs(server_ctx, test_dict.get('handle_stop'))

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

        with self.assertRaises(TypeError) as cm:
            api.operation()(42)
        self.assertEqual("API datasets: @operation() decorator"
                         " must be used with one of the HTTP"
                         " methods of an ApiHandler",
                         f"{cm.exception}")


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
    def test_basic_props(self):
        config = {}
        server_ctx = ServerContext(mock_server(), config)
        api_ctx = ApiContext(server_ctx)
        self.assertIsInstance(api_ctx.config, FrozenDict)
        self.assertEqual(config, api_ctx.config)
        self.assertEqual((), api_ctx.apis)
        self.assertIsInstance(api_ctx.get_open_api_doc(), dict)

    def test_async_exec(self):
        framework = MockFramework()
        server = mock_server(framework=framework)
        server_ctx = ServerContext(server, {})
        api_ctx = ApiContext(server_ctx)

        def my_func(a, b):
            return a + b

        self.assertEqual(0, framework.call_later_count)
        api_ctx.call_later(0.1, my_func, 40, 2)
        self.assertEqual(1, framework.call_later_count)

        self.assertEqual(0, framework.run_in_executor_count)
        api_ctx.run_in_executor(None, my_func, 40, 2)
        self.assertEqual(1, framework.run_in_executor_count)


class ApiHandlerTest(unittest.TestCase):
    class DatasetsContext(ApiContext):

        def on_update(self, prev_ctx: Optional[Context]):
            pass

    def setUp(self) -> None:
        self.api = Api("datasets", create_ctx=self.DatasetsContext)
        self.config = {}
        server_ctx = ServerContext(mock_server(api_specs=[self.api]),
                                   self.config)
        server_ctx.on_update(None)
        self.request = MockApiRequest()
        self.response = MockApiResponse()
        self.handler = ApiHandler(server_ctx.get_api_ctx("datasets"),
                                  self.request,
                                  self.response)

    def test_props(self):
        handler = self.handler
        self.assertIs(self.request, handler.request)
        self.assertIs(self.response, handler.response)
        self.assertIsInstance(handler.ctx, self.DatasetsContext)
        self.assertIsInstance(handler.ctx.config, FrozenDict)
        self.assertEqual(self.config, handler.ctx.config)

    def test_default_methods(self):
        handler = self.handler

        with self.assertRaises(ApiError.MethodNotAllowed) as cm:
            handler.get()
        self.assertEqual('HTTP status 405: method GET not allowed',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.MethodNotAllowed) as cm:
            handler.post()
        self.assertEqual('HTTP status 405: method POST not allowed',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.MethodNotAllowed) as cm:
            handler.put()
        self.assertEqual('HTTP status 405: method PUT not allowed',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.MethodNotAllowed) as cm:
            handler.delete()
        self.assertEqual('HTTP status 405: method DELETE not allowed',
                         f'{cm.exception}')

    def test_default_options_method(self):
        handler = self.handler
        # Should not raise
        handler.options()


class ApiRequestTest(unittest.TestCase):

    def test_urls(self):
        request = MockApiRequest(dict(details='1'),
                                 reverse_url_prefix='/proxy/9192')
        self.assertEqual('http://localhost:8080/datasets?details=1',
                         request.url)
        self.assertEqual('http://localhost:8080',
                         request.base_url)
        self.assertEqual('http://localhost:8080/proxy/9192',
                         request.reverse_base_url)
        self.assertEqual('http://localhost:8080/places',
                         request.url_for_path('places'))
        self.assertEqual('http://localhost:8080/proxy/9192/places',
                         request.url_for_path('places', reverse=True))

    def test_query_args(self):
        request = MockApiRequest(query_args=dict(details=['1']))
        self.assertEqual(['1'], request.get_query_args('details'))
        self.assertEqual([], request.get_query_args('crs'))

    def test_query_args_with_type(self):
        request = MockApiRequest(query_args=dict(details=['1']))
        self.assertEqual([True], request.get_query_args('details', type=bool))
        self.assertEqual([], request.get_query_args('crs', type=str))

    def test_query_arg_with_type(self):
        request = MockApiRequest(query_args=dict(details=['1']))
        self.assertEqual(True, request.get_query_arg('details', type=bool))
        self.assertEqual(None, request.get_query_arg('crs', type=str))

    def test_query_arg_with_default(self):
        request = MockApiRequest(query_args=dict(details=['1']))
        self.assertEqual(True, request.get_query_arg('details',
                                                     default=False))
        self.assertEqual('CRS84', request.get_query_arg('crs',
                                                        default='CRS84'))


class ApiErrorTest(unittest.TestCase):

    def test_base_class(self):
        error = ApiError(428)
        self.assertIsInstance(error, Exception)
        self.assertEqual(428, error.status_code)
        self.assertEqual(None, error.message)

        error = ApiError(428, message='No idea')
        self.assertIsInstance(error, Exception)
        self.assertEqual(428, error.status_code)
        self.assertEqual('No idea', error.message)

        self.assertIsNot(ApiError.BadRequest, ApiError.Unauthorized)
        self.assertIsNot(ApiError.BadRequest, ApiError.Forbidden)

    def test_bad_request(self):
        error = ApiError.BadRequest()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(400, error.status_code)

    def test_unauthorized(self):
        error = ApiError.Unauthorized()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(401, error.status_code)

    def test_forbidden(self):
        error = ApiError.Forbidden()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(403, error.status_code)

    def test_not_found(self):
        error = ApiError.NotFound()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(404, error.status_code)

    def test_method_not_allowed(self):
        error = ApiError.MethodNotAllowed()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(405, error.status_code)

    def test_conflict(self):
        error = ApiError.Conflict()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(409, error.status_code)

    def test_gone(self):
        error = ApiError.Gone()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(410, error.status_code)

    def test_not_implemented(self):
        error = ApiError.NotImplemented()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(501, error.status_code)

    def test_invalid_server_config(self):
        error = ApiError.InvalidServerConfig()
        self.assertIsInstance(error, ApiError)
        self.assertEqual(580, error.status_code)
