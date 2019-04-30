# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from json import JSONDecodeError
from typing import Optional, Any, Dict

import tornado.escape
import tornado.options
import yaml
from tornado.ioloop import IOLoop
from tornado.log import enable_pretty_logging
from tornado.web import RequestHandler, Application

from .context import ServiceContext
from .defaults import DEFAULT_ADDRESS, DEFAULT_PORT, DEFAULT_CONFIG_FILE, DEFAULT_UPDATE_PERIOD, DEFAULT_LOG_PREFIX, \
    DEFAULT_TILE_CACHE_SIZE, DEFAULT_NAME, DEFAULT_TRACE_PERF, DEFAULT_TILE_COMP_MODE
from .errors import ServiceBadRequestError
from .reqparams import RequestParams
from .undefined import UNDEFINED

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

_LOG = logging.getLogger('xcube')


class Service:
    """
    A web service that provides a remote API to some application.
    """

    def __init__(self,
                 application: Application,
                 name: str = DEFAULT_NAME,
                 address: str = DEFAULT_ADDRESS,
                 port: int = DEFAULT_PORT,
                 config_file: Optional[str] = None,
                 tile_cache_size: Optional[str] = DEFAULT_TILE_CACHE_SIZE,
                 tile_comp_mode: int = DEFAULT_TILE_COMP_MODE,
                 update_period: Optional[float] = DEFAULT_UPDATE_PERIOD,
                 trace_perf: bool = DEFAULT_TRACE_PERF,
                 log_file_prefix: str = DEFAULT_LOG_PREFIX,
                 log_to_stderr: bool = False) -> None:

        """
        Start a tile service.

        The *service_info_file*, if given, represents the service in the filesystem, similar to
        the ``/var/run/`` directory on Linux systems.

        If the service file exist and its information is compatible with the requested *port*, *address*, *caller*, then
        this function simply returns without taking any other actions.

        :param application: The Tornado web application
        :param address: the address
        :param port: the port number
        :param config_file: optional configuration file
        :param update_period: if not-None, time of idleness in seconds before service is updated
        :param log_file_prefix: Log file prefix, default is "xcube_server.log"
        :param log_to_stderr: Whether logging should be shown on stderr
        :return: service information dictionary
        """
        log_dir = os.path.dirname(log_file_prefix)
        if log_dir and not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        options = tornado.options.options
        options.log_file_prefix = log_file_prefix or 'xcube_server.log'
        options.log_to_stderr = log_to_stderr
        enable_pretty_logging()

        tile_cache_config = parse_tile_cache_config(tile_cache_size)

        self.config_file = os.path.abspath(config_file) if config_file else None
        self.config_mtime = None
        self.update_period = update_period
        self.update_timer = None
        self.config_error = None
        self.service_info = dict(port=port,
                                 address=address,
                                 started=datetime.now().isoformat(sep=' '),
                                 pid=os.getpid())

        self.context = ServiceContext(name=name,
                                      base_dir=os.path.dirname(self.config_file or os.path.abspath('')),
                                      tile_comp_mode=tile_comp_mode,
                                      trace_perf=trace_perf,
                                      mem_tile_cache_capacity=tile_cache_config.get("capacity"))
        self._maybe_load_config()

        application.service_context = self.context
        application.time_of_last_activity = time.process_time()
        self.application = application

        self.server = application.listen(port, address=address or 'localhost')
        # Ensure we have the same event loop in all threads
        asyncio.set_event_loop_policy(_GlobalEventLoopPolicy(asyncio.get_event_loop()))
        # Register handlers for common termination signals
        signal.signal(signal.SIGINT, self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)
        self._maybe_load_config()
        self._maybe_install_update_check()

    def start(self):
        address = self.service_info['address']
        port = self.service_info['port']
        test_url = self.context.get_service_url(f"http://{address}:{port}", "datasets")
        _LOG.info(f'service running, listening on {address}:{port}, try {test_url}')
        _LOG.info(f'press CTRL+C to stop service')
        if len(self.context.config.get('Datasets', {})) == 0:
            _LOG.warning('no datasets configured')
        IOLoop.current().start()

    def stop(self, kill=False):
        """
        Stops the Tornado web server.
        """
        if kill:
            sys.exit(0)
        else:
            IOLoop.current().add_callback(self._on_shut_down)

    def _on_shut_down(self):

        _LOG.info('stopping service...')

        # noinspection PyUnresolvedReferences,PyBroadException
        try:
            self.update_timer.cancel()
        except Exception:
            pass

        if self.server:
            self.server.stop()
            self.server = None

        IOLoop.current().stop()

    # noinspection PyUnusedLocal
    def _sig_handler(self, sig, frame):
        _LOG.warning(f'caught signal {sig}')
        IOLoop.current().add_callback_from_signal(self._on_shut_down)

    def _maybe_install_update_check(self):
        if self.update_period is None or self.update_period <= 0:
            return
        IOLoop.current().call_later(self.update_period, self._maybe_check_for_updates)

    def _maybe_check_for_updates(self):
        self._maybe_load_config()
        self._maybe_install_update_check()

    def _maybe_load_config(self):
        config_file = self.config_file
        if config_file is None:
            config_file = DEFAULT_CONFIG_FILE
        try:
            stat = os.stat(config_file)
        except OSError as e:
            if self.config_error is None:
                _LOG.error(f'configuration file {config_file!r}: {e}')
                self.config_error = e
            return
        if self.config_mtime != stat.st_mtime:
            self.config_mtime = stat.st_mtime
            try:
                with open(config_file) as stream:
                    self.context.config = yaml.safe_load(stream)
                self.config_error = None
                _LOG.info(f'configuration file {config_file!r} successfully loaded')
            except (yaml.YAMLError, OSError) as e:
                if self.config_error is None:
                    _LOG.error(f'configuration file {config_file!r}: {e}')
                    self.config_error = e
                return


# noinspection PyAbstractClass
class ServiceRequestHandler(RequestHandler):

    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)
        self._params = ServiceRequestParams(self)

    @property
    def service_context(self) -> ServiceContext:
        return self.application.service_context

    @property
    def base_url(self):
        return self.request.protocol + '://' + self.request.host

    @property
    def params(self) -> 'ServiceRequestParams':
        return self._params

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET, PUT, DELETE, OPTIONS')

    def options(self):
        self.set_status(204)
        self.finish()

    def get_body_as_json_object(self, name="JSON object"):
        """ Get the body argument as JSON object. """
        try:
            return tornado.escape.json_decode(self.request.body)
        except (JSONDecodeError, TypeError, ValueError) as e:
            raise ServiceBadRequestError(f"Invalid or missing {name} in request body") from e

    def on_finish(self):
        """
        Store time of last activity so we can measure time of inactivity and then optionally auto-exit.
        """
        self.application.time_of_last_activity = time.process_time()

    def write_error(self, status_code, **kwargs):
        self.set_header('Content-Type', 'application/json')
        # if self.settings.get("serve_traceback") and "exc_info" in kwargs:
        if "exc_info" in kwargs:
            # in debug mode, try to send a traceback
            lines = []
            for line in traceback.format_exception(*kwargs["exc_info"]):
                lines.append(line)
            self.finish(json.dumps({
                'error': {
                    'code': status_code,
                    'message': self._reason,
                    'traceback': lines,
                }
            }, indent=2))
        else:
            self.finish(json.dumps({
                'error': {
                    'code': status_code,
                    'message': self._reason,
                }
            }, indent=2))


class ServiceRequestParams(RequestParams):
    def __init__(self, handler: RequestHandler):
        self.handler = handler

    def get_query_argument(self, name: str, default: Optional[str] = UNDEFINED) -> Optional[str]:
        """
        Get query argument.
        :param name: Query argument name
        :param default: Default value.
        :return: the value or none
        :raise: ServiceBadRequestError
        """
        value = self.handler.get_query_argument(name, default=default)
        if value is UNDEFINED:
            raise ServiceBadRequestError(f'Missing query parameter "{name}"')
        return value


# noinspection PyAbstractClass
class _GlobalEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """
    Event loop policy that has one fixed global loop for all threads.

    We use it for the following reason: As of Tornado 5 IOLoop.current() no longer has
    a single global instance. It is a thread-local instance, but only on the main thread.
    Other threads have no IOLoop instance by default.

    _GlobalEventLoopPolicy is a fix that allows us to access the same IOLoop
    in all threads.

    Usage::

        asyncio.set_event_loop_policy(_GlobalEventLoopPolicy(asyncio.get_event_loop()))

    """

    def __init__(self, global_loop):
        super().__init__()
        self._global_loop = global_loop

    def get_event_loop(self):
        return self._global_loop


def url_pattern(pattern: str):
    """
    Convert a string *pattern* where any occurrences of ``{{NAME}}`` are replaced by an equivalent
    regex expression which will assign matching character groups to NAME. Characters match until
    one of the RFC 2396 reserved characters is found or the end of the *pattern* is reached.

    The function can be used to map URLs patterns to request handlers as desired by the Tornado web server, see
    http://www.tornadoweb.org/en/stable/web.html

    RFC 2396 Uniform Resource Identifiers (URI): Generic Syntax lists
    the following reserved characters::

        reserved    = ";" | "/" | "?" | ":" | "@" | "&" | "=" | "+" | "$" | ","

    :param pattern: URL pattern
    :return: equivalent regex pattern
    :raise ValueError: if *pattern* is invalid
    """
    name_pattern = '(?P<%s>[^\;\/\?\:\@\&\=\+\$\,]+)'
    reg_expr = ''
    pos = 0
    while True:
        pos1 = pattern.find('{{', pos)
        if pos1 >= 0:
            pos2 = pattern.find('}}', pos1 + 2)
            if pos2 > pos1:
                name = pattern[pos1 + 2:pos2]
                if not name.isidentifier():
                    raise ValueError('name in {{name}} must be a valid identifier, but got "%s"' % name)
                reg_expr += pattern[pos:pos1] + (name_pattern % name)
                pos = pos2 + 2
            else:
                raise ValueError('no matching "}}" after "{{" in "%s"' % pattern)

        else:
            reg_expr += pattern[pos:]
            break
    return reg_expr


def parse_tile_cache_config(tile_cache_size: str) -> Dict[str, Any]:
    tile_cache_size = tile_cache_size.upper()
    if tile_cache_size != "" and tile_cache_size != "OFF":
        unit = tile_cache_size[-1]
        factors = {"B": 10 ** 0, "K": 10 ** 3, "M": 10 ** 6, "G": 10 ** 9, "T": 10 ** 12}
        try:
            if unit in factors:
                capacity = int(tile_cache_size[0: -1]) * factors[unit]
            else:
                capacity = int(tile_cache_size)
        except ValueError:
            raise ValueError(f"invalid tile cache size: {tile_cache_size!r}")
        if capacity > 0:
            return dict(capacity=capacity)
        elif capacity < 0:
            raise ValueError(f"negative tile cache size: {tile_cache_size!r}")
    return dict(no_cache=True)
