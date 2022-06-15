#  The MIT License (MIT)
#  Copyright (c) 2022 by the xcube development team and contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

import unittest
from typing import Sequence, Optional, Callable, Any, Awaitable, Union, Tuple

import pytest
import tornado.httputil
import tornado.web
from tornado import concurrent

from test.server.mocks import mock_server
from xcube.server.api import ApiError, ServerConfig, Api
from xcube.server.api import ApiHandler
from xcube.server.api import ApiRoute
from xcube.server.api import Context
from xcube.server.asyncexec import ReturnT
from xcube.server.webservers.tornado import SERVER_CTX_ATTR_NAME
from xcube.server.webservers.tornado import TornadoApiRequest
from xcube.server.webservers.tornado import TornadoFramework
from xcube.server.webservers.tornado import TornadoRequestHandler


class TornadoFrameworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.application = MockApplication()
        self.io_loop = MockIOLoop()
        # noinspection PyTypeChecker
        self.framework = TornadoFramework(application=self.application,
                                          io_loop=self.io_loop)

    # def test_add_routes(self):
    #     self.framework.add_routes([ApiRoute()])

    def test_start_and_update_and_stop(self):
        server = mock_server(framework=self.framework, config={})
        self.assertEqual(0, self.application.listen_count)
        self.assertEqual(0, self.io_loop.start_count)
        self.assertEqual(0, self.io_loop.stop_count)
        self.framework.start(server.ctx)
        self.assertEqual(1, self.application.listen_count)
        self.assertEqual(1, self.io_loop.start_count)
        self.assertEqual(0, self.io_loop.stop_count)
        self.framework.update(server.ctx)
        self.assertEqual(1, self.application.listen_count)
        self.assertEqual(1, self.io_loop.start_count)
        self.assertEqual(0, self.io_loop.stop_count)
        self.framework.stop(server.ctx)
        self.assertEqual(1, self.application.listen_count)
        self.assertEqual(1, self.io_loop.start_count)
        self.assertEqual(1, self.io_loop.stop_count)

    def test_async_exec(self):
        def my_func(a, b):
            return a + b

        self.assertEqual(0, self.io_loop.call_later_count)
        self.framework.call_later(0.1, my_func, 40, 2)
        self.assertEqual(1, self.io_loop.call_later_count)

        self.assertEqual(0, self.io_loop.run_in_executor_count)
        self.framework.run_in_executor(None, my_func, 40, 2)
        self.assertEqual(1, self.io_loop.run_in_executor_count)

    def test_path_to_pattern_ok(self):
        p2p = TornadoFramework.path_to_pattern

        self.assertEqual("/collections",
                         p2p("/collections"))

        self.assertEqual("/collections/"
                         "("
                         "?P<collection_id>"
                         "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
                         ")",
                         p2p("/collections/{collection_id}"))

        self.assertEqual("/collections/"
                         "("
                         "?P<collection_id>"
                         "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
                         ")"
                         "/items",
                         p2p("/collections/{collection_id}/items"))

        self.assertEqual("/collections/"
                         "("
                         "?P<collection_id>"
                         "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
                         ")"
                         "/items/"
                         "("
                         "?P<item_id>"
                         "[^\\;\\/\\?\\:\\@\\&\\=\\+\\$\\,]+"
                         ")",
                         p2p("/collections/{collection_id}/items/{item_id}"))

    def test_path_to_pattern_fails(self):
        p2p = TornadoFramework.path_to_pattern

        with self.assertRaises(ValueError) as cm:
            p2p('/datasets/{dataset id}')
        self.assertEqual('"{name}" in path must be a valid identifier,'
                         ' but got "dataset id"',
                         f"{cm.exception}")

        with self.assertRaises(ValueError) as cm:
            p2p('/datasets/{dataset_id')
        self.assertEqual('missing closing "}"'
                         ' in "/datasets/{dataset_id"',
                         f"{cm.exception}")

        with self.assertRaises(ValueError) as cm:
            p2p('/datasets/dataset_id}/bbox')
        self.assertEqual('missing opening "{"'
                         ' in "/datasets/dataset_id}/bbox"',
                         f"{cm.exception}")


# class TornadoRequestHandlerTest(unittest.TestCase):
#     def test_basic_props(self):
#         TornadoRequestHandler()


class TornadoApiRequestTest(unittest.TestCase):
    def test_basic_props(self):
        body = b'{"type": "feature", "id": 137}'
        tr = tornado.httputil.HTTPServerRequest(
            method='GET',
            host='localhost:8080',
            uri='/datasets',
            body=body
        )
        request = TornadoApiRequest(tr)
        self.assertEqual("http://localhost:8080/datasets", request.url)
        self.assertEqual(body, request.body)
        self.assertEqual({"type": "feature", "id": 137}, request.json)
        self.assertEqual([],
                         request.get_query_args('details'))

    def test_url_for_path(self):
        tr = tornado.httputil.HTTPServerRequest(
            method='GET',
            host='localhost:8080',
            uri='/datasets?details=1',
        )
        request = TornadoApiRequest(tr)
        self.assertEqual("http://localhost:8080/collections",
                         request.url_for_path('collections'))
        self.assertEqual("http://localhost:8080/collections?details=1",
                         request.url_for_path('/collections',
                                              query='details=1'))

    def test_get_query_args(self):
        tr = tornado.httputil.HTTPServerRequest(
            method='GET',
            host='localhost:8080',
            uri='/datasets?details=1',
        )
        request = TornadoApiRequest(tr)
        self.assertEqual(['1'],
                         request.get_query_args('details'))
        self.assertEqual(['1'],
                         request.get_query_args('details', type=str))
        self.assertEqual([1],
                         request.get_query_args('details', type=int))
        self.assertEqual([True],
                         request.get_query_args('details', type=bool))

    def test_get_query_args_invalid_type(self):
        tr = tornado.httputil.HTTPServerRequest(
            method='GET',
            host='localhost:8080',
            uri='/datasets?details=x',
        )
        request = TornadoApiRequest(tr)
        with self.assertRaises(tornado.web.HTTPError) as cm:
            request.get_query_args('details', type=int)
        self.assertEqual("HTTP 400: Bad Request"
                         " (Query parameter 'details'"
                         " must have type 'int'.)",
                         f'{cm.exception}')


class TornadoRequestHandlerTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_api_error_converted(self):
        application = tornado.web.Application()

        setattr(application, SERVER_CTX_ATTR_NAME, MockContext())

        # noinspection PyTypeChecker
        request = tornado.httputil.HTTPServerRequest(
            method='GET',
            host='localhost:8080',
            uri='/datasets?details=x',
            connection=MockConnection()
        )

        class TestHandler(ApiHandler):
            def get(self):
                raise ApiError(550)

            def post(self):
                raise ApiError(551)

            def put(self):
                raise ApiError(552)

            def delete(self):
                raise ApiError(553)

            def options(self):
                raise ApiError(554)

        api_route = ApiRoute('test', '/test', TestHandler)
        handler = TornadoRequestHandler(application, request,
                                        api_route=api_route)

        async def test_it():
            with pytest.raises(tornado.web.HTTPError, match=r".*550.*"):
                await handler.get()
            with pytest.raises(tornado.web.HTTPError, match=r".*551.*"):
                await handler.post()
            with pytest.raises(tornado.web.HTTPError, match=r".*552.*"):
                await handler.put()
            with pytest.raises(tornado.web.HTTPError, match=r".*553.*"):
                await handler.delete()
            with pytest.raises(tornado.web.HTTPError, match=r".*554.*"):
                await handler.options()

        import asyncio
        asyncio.run(test_it())


# Helpers

class MockIOLoop:
    def __init__(self):
        self.start_count = 0
        self.stop_count = 0
        self.call_later_count = 0
        self.run_in_executor_count = 0

    def start(self):
        self.start_count += 1

    def stop(self):
        self.stop_count += 1

    def call_later(self, *args, **kwargs):
        self.call_later_count += 1

    def run_in_executor(self, *args, **kwargs):
        self.run_in_executor_count += 1


class MockApplication:
    def __init__(self):
        self.handlers = []
        self.listen_count = 0

    def add_handlers(self, domain: str, handlers: Sequence):
        self.handlers.extend(handlers)

    def listen(self, *args, **kwargs):
        self.listen_count += 1


class MockConnection:
    def __init__(self):
        self._cb = None

    def set_close_callback(self, cb):
        self._cb = cb


class MockContext(Context):

    def __init__(self):
        self._api = Api('test')
        self._config = {}

    @property
    def apis(self) -> Tuple[Api]:
        return self._api,

    @property
    def config(self) -> ServerConfig:
        return self.config

    def get_api_ctx(self, api_name: str) -> Optional["Context"]:
        return self._api if api_name == 'test' else None

    def on_update(self, prev_context: Optional["Context"]):
        pass

    def on_dispose(self):
        pass

    def call_later(self, delay: Union[int, float], callback: Callable,
                   *args, **kwargs) -> object:
        pass

    def run_in_executor(self,
                        executor: Optional[concurrent.futures.Executor],
                        function: Callable[..., ReturnT], *args: Any,
                        **kwargs: Any) -> Awaitable[ReturnT]:
        pass
