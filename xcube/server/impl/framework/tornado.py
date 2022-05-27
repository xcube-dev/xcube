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

import logging
from abc import ABC
from typing import Any, Optional, Sequence, Union, Callable

import tornado.escape
import tornado.httputil
import tornado.ioloop
import tornado.web

from xcube.server.api import ApiHandler
from xcube.server.api import ApiRequest
from xcube.server.api import ApiResponse
from xcube.server.api import ApiRoute
from xcube.server.api import JSON
from xcube.server.context import Context
from xcube.server.framework import ServerFramework

_CTX_ATTR_NAME = "__xcube_server_root_context"


class TornadoFramework(ServerFramework):
    """
    The Tornado web server framework.
    """

    def __init__(self,
                 application: Optional[tornado.web.Application] = None,
                 io_loop: Optional[tornado.ioloop.IOLoop] = None):
        self._application = application or tornado.web.Application()
        self._io_loop = io_loop or tornado.ioloop.IOLoop.current()
        self._configure_logger()

    def add_routes(self, api_routes: Sequence[ApiRoute]):
        handlers = []
        for api_route in api_routes:
            # noinspection PyAbstractClass
            class TornadoHandler(TornadoBaseHandler):
                pass

            handlers.append((
                api_route.pattern,
                TornadoHandler,
                {
                    "api_route": api_route
                }
            ))

        self._application.add_handlers(".*$", handlers)

    def update(self, ctx: Context):
        setattr(self._application, _CTX_ATTR_NAME, ctx)

    def start(self, ctx: Context):
        config = ctx.config
        port = config["port"]
        address = config["address"]
        self._application.listen(port, address=address)
        self._io_loop.start()

    def stop(self, ctx: Context):
        self._io_loop.stop()

    def call_later(self,
                   delay: Union[int, float],
                   callback: Callable,
                   *args,
                   **kwargs):
        self._io_loop.call_later(delay, callback, *args, **kwargs)

    @staticmethod
    def _configure_logger():
        # Configure Tornado logger to use configured root handlers.
        # For some reason, Tornado's log records will not arrive at
        # the root logger.
        tornado_logger = logging.getLogger('tornado')
        for h in list(tornado_logger.handlers):
            tornado_logger.removeHandler(h)
            h.close()
        for h in list(logging.root.handlers):
            tornado_logger.addHandler(h)
        tornado_logger.setLevel(logging.root.level)


class TornadoBaseHandler(tornado.web.RequestHandler, ABC):

    def __init__(self,
                 application: tornado.web.Application,
                 request: tornado.httputil.HTTPServerRequest,
                 **kwargs: Any):
        super().__init__(application, request)
        root_ctx = getattr(application, _CTX_ATTR_NAME, None)
        assert isinstance(root_ctx, Context)
        api_route: ApiRoute = kwargs.pop("api_route")
        self._api_handler: ApiHandler = api_route.handler_cls(
            api_route.api_name,
            root_ctx,
            TornadoApiRequest(request),
            TornadoApiResponse(self),
            **api_route.handler_kwargs
        )

    def write_error(self, status_code: int, **kwargs: Any):
        self.finish({
            "error": {
                "status_code": status_code,
                **kwargs
            }
        })

    async def get(self, *args, **kwargs):
        return self._api_handler.get(*args, **kwargs)

    async def post(self, *args, **kwargs):
        return self._api_handler.post(*args, **kwargs)

    async def put(self, *args, **kwargs):
        return self._api_handler.put(*args, **kwargs)

    async def delete(self, *args, **kwargs):
        return self._api_handler.delete(*args, **kwargs)

    async def options(self, *args, **kwargs):
        return self._api_handler.options(*args, **kwargs)


class TornadoApiRequest(ApiRequest):
    def __init__(self, request: tornado.httputil.HTTPServerRequest):
        self._request = request

    def get_query_args(self, name: str) -> Sequence[str]:
        return self._request.query_arguments.get(name, [])

    def get_body_args(self, name: str) -> Sequence[bytes]:
        return self._request.body_arguments.get(name, [])

    @property
    def body(self) -> bytes:
        return self._request.body

    @property
    def json(self) -> JSON:
        return tornado.escape.json_decode(self.body)


class TornadoApiResponse(ApiResponse):
    def __init__(self, handler: TornadoBaseHandler):
        self._handler = handler

    def write(self, data: Union[str, bytes, JSON]):
        self._handler.write(data)

    def finish(self, data: Union[str, bytes, JSON] = None):
        self._handler.finish(data)

    def error(self,
              status_code: int,
              message: Optional[str] = None,
              *args: Any,
              **kwargs: Any) -> Exception:
        return tornado.web.HTTPError(status_code,
                                     log_message=message,
                                     *args,
                                     **kwargs)
