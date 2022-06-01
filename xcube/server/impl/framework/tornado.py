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

import concurrent.futures
import functools
import logging
import traceback
import urllib.parse
from abc import ABC
from typing import Any, Optional, Sequence, Union, Callable, Dict, Type, \
    Awaitable

import tornado.escape
import tornado.httputil
import tornado.ioloop
import tornado.web

from xcube.constants import LOG
from xcube.constants import LOG_LEVEL_DETAIL
from xcube.server.api import ApiHandler, ArgT
from xcube.server.api import ApiRequest
from xcube.server.api import ApiResponse
from xcube.server.api import ApiRoute
from xcube.server.api import Context
from xcube.server.api import JSON
from xcube.server.api import ReturnT
from xcube.server.framework import ServerFramework
from xcube.util.assertions import assert_true
from xcube.version import version

_CTX_ATTR_NAME = "__xcube_server_root_context"


class TornadoFramework(ServerFramework):
    """
    The Tornado web server framework.
    """

    def __init__(self,
                 application: Optional[tornado.web.Application] = None,
                 io_loop: Optional[tornado.ioloop.IOLoop] = None):
        self._application = application or tornado.web.Application()
        self._io_loop = io_loop
        self._configure_logger()

    @property
    def io_loop(self) -> tornado.ioloop.IOLoop:
        return self._io_loop or tornado.ioloop.IOLoop.current()

    def add_routes(self, api_routes: Sequence[ApiRoute]):
        handlers = []
        for api_route in api_routes:
            # noinspection PyAbstractClass
            class TornadoHandler(TornadoBaseHandler):
                pass

            handlers.append((
                self._convert_path_to_pattern(api_route.path),
                TornadoHandler,
                {
                    "api_route": api_route
                }
            ))

            LOG.log(LOG_LEVEL_DETAIL, f'Added route'
                                      f' {api_route.path!r}'
                                      f' from API {api_route.api_name!r}')

        self._application.add_handlers(".*$", handlers)

    def update(self, ctx: Context):
        setattr(self._application, _CTX_ATTR_NAME, ctx)

    def start(self, ctx: Context):
        config = ctx.config

        port = config["port"]
        address = config["address"]

        self._application.listen(port, address=address)

        address_ = "127.0.0.1" if address == "0.0.0.0" else address
        test_url = f"http://{address_}:{port}/openapi"
        LOG.info(f"Service running, listening on {address}:{port}")
        LOG.info(f"Try {test_url}")
        LOG.info(f"Press CTRL+C to stop service")

        self.io_loop.start()

    def stop(self, ctx: Context):
        self.io_loop.stop()

    def call_later(self,
                   delay: Union[int, float],
                   callback: Callable,
                   *args,
                   **kwargs) -> object:
        return self.io_loop.call_later(
            delay,
            callback,
            *args,
            **kwargs
        )

    def run_in_executor(
            self,
            executor: Optional[concurrent.futures.Executor],
            function: Callable[..., ReturnT],
            *args: Any,
            **kwargs: Any
    ) -> Awaitable[ReturnT]:
        return self.io_loop.run_in_executor(
            executor,
            functools.partial(function, **kwargs) if kwargs else function,
            *args
        )

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

    @staticmethod
    def _convert_path_to_pattern(path: str):
        """
        Convert a string *pattern* where any occurrences of ``{NAME}``
        are replaced by an equivalent regex expression which will
        assign matching character groups to NAME. Characters match until
        one of the RFC 2396 reserved characters is found or the end of
        the *pattern* is reached.

        :param path: URL path
        :return: equivalent regex pattern
        :raise ValueError: if *pattern* is invalid
        """
        if '{' not in path:
            return path
        name_pattern = r'(?P<%s>[^\;\/\?\:\@\&\=\+\$\,]+)'
        reg_expr = ''
        pos = 0
        while True:
            pos1 = path.find('{', pos)
            if pos1 >= 0:
                pos2 = path.find('}', pos1 + 1)
                if pos2 > pos1:
                    name = path[pos1 + 1:pos2]
                    if not name.isidentifier():
                        raise ValueError(
                            '"{name}" in path must be a valid identifier,'
                            ' but got "%s"' % name)
                    reg_expr += path[pos:pos1] + (name_pattern % name)
                    pos = pos2 + 2
                else:
                    raise ValueError('no matching "}"'
                                     ' after "{" in "%s"' % path)
            else:
                reg_expr += path[pos:]
                break
        return reg_expr


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

    def set_default_headers(self):
        self.set_header('Server', f'xcube-server/{version}')
        # TODO: get from config
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods',
                        'GET,PUT,DELETE,OPTIONS')
        self.set_header('Access-Control-Allow-Headers',
                        'x-requested-with,access-control-allow-origin,'
                        'authorization,content-type')

    def write_error(self, status_code: int, **kwargs: Any):
        valid_json_types = str, int, float, bool, type(None)
        error_info = {k: v for k, v in kwargs.items()
                      if isinstance(v, valid_json_types)}
        error_info.update(status_code=status_code)
        if "exc_info" in kwargs:
            exc_type, exc_val, exc_tb = kwargs["exc_info"]
            error_info.update(
                exception=traceback.format_exception(exc_type, exc_val,
                                                     exc_tb),
                message=str(exc_val)
            )
            if isinstance(exc_val, tornado.web.HTTPError) and exc_val.reason:
                error_info.update(reason=exc_val.reason)
        self.finish({
            "error": error_info
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

    # # noinspection PyUnusedLocal
    # def options(self, *args, **kwargs):
    #     self.set_status(204)
    #     self.finish()


class TornadoApiRequest(ApiRequest):
    def __init__(self, request: tornado.httputil.HTTPServerRequest):
        self._request = request
        self._query_args = None

    @functools.cached_property
    def query_args(self) -> Dict[str, Sequence[str]]:
        return urllib.parse.parse_qs(self._request.query)

    # noinspection PyShadowingBuiltins
    def get_query_args(self,
                       name: str,
                       type: Optional[Type[ArgT]] = None) -> Sequence[ArgT]:
        query_args = self.query_args
        if not query_args or name not in query_args:
            return []
        values = query_args[name]
        if type is not None and type is not str:
            assert_true(callable(type), 'type must be callable')
            try:
                return [type(v) for v in values]
            except (ValueError, TypeError):
                raise tornado.web.HTTPError(
                    400, f'Query parameter {name!r}'
                         f' must have type {type.__name__!r}.'
                )
        return values

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

    def set_status(self, status_code: int, reason: Optional[str] = None):
        self._handler.set_status(status_code, reason=reason)

    def write(self, data: Union[str, bytes, JSON]):
        self._handler.write(data)

    def finish(self, data: Union[str, bytes, JSON] = None):
        self._handler.finish(data)

    def error(self,
              status_code: int,
              message: Optional[str] = None,
              reason: Optional[str] = None) -> Exception:
        return tornado.web.HTTPError(status_code,
                                     log_message=message,
                                     reason=reason)
