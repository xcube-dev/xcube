# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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
import configparser
import json
import os
import os.path
import signal
import sys
import time
import traceback
from datetime import datetime
from json import JSONDecodeError
from typing import Optional, Dict, List, Tuple, Mapping

import tornado.escape
import tornado.options
from tornado.ioloop import IOLoop
from tornado.log import enable_pretty_logging
from tornado.web import Application
from tornado.web import RequestHandler

from xcube.constants import LOG
from xcube.core.dsio import is_s3_url
from xcube.core.mldataset import guess_ml_dataset_format
from xcube.util.cache import parse_mem_size
from xcube.util.caseless import caseless_dict
from xcube.util.config import load_configs
from xcube.util.config import load_json_or_yaml_config
from xcube.util.undefined import UNDEFINED
from xcube.version import version
from xcube.webapi.context import ServiceContext
from xcube.webapi.defaults import DEFAULT_ADDRESS
from xcube.webapi.defaults import DEFAULT_LOG_PREFIX
from xcube.webapi.defaults import DEFAULT_PORT
from xcube.webapi.defaults import DEFAULT_TILE_CACHE_SIZE
from xcube.webapi.defaults import DEFAULT_TILE_COMP_MODE
from xcube.webapi.defaults import DEFAULT_TRACE_PERF
from xcube.webapi.defaults import DEFAULT_UPDATE_PERIOD
from xcube.webapi.errors import ServiceBadRequestError
from xcube.webapi.reqparams import RequestParams

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

SNAP_CPD_LIST = list()


class Service:
    """
    A web service that provides a remote API to some application.
    """

    def __init__(self,
                 application: Application,
                 prefix: str = None,
                 address: str = DEFAULT_ADDRESS,
                 port: int = DEFAULT_PORT,
                 cube_paths: List[str] = None,
                 styles: Dict[str, Tuple] = None,
                 config_file: Optional[str] = None,
                 base_dir: Optional[str] = None,
                 tile_cache_size: Optional[str] = DEFAULT_TILE_CACHE_SIZE,
                 tile_comp_mode: int = DEFAULT_TILE_COMP_MODE,
                 update_period: Optional[float] = DEFAULT_UPDATE_PERIOD,
                 trace_perf: bool = DEFAULT_TRACE_PERF,
                 log_file_prefix: str = DEFAULT_LOG_PREFIX,
                 log_to_stderr: bool = False,
                 aws_prof: str = None,
                 aws_env: bool = False) -> None:

        """
        Start a tile service.

        The *service_info_file*, if given, represents the service in the filesystem, similar to
        the ``/var/run/`` directory on Linux systems.

        If the service file exist and its information is compatible with the requested *port*, *address*, *caller*, then
        this function simply returns without taking any other actions.

        :param application: The Tornado web application
        :param address: the address
        :param port: the port number
        :param cube_paths: optional list of cube paths
        :param config_file: optional configuration file
        :param base_dir: optional base directory
        :param update_period: if not-None, time of idleness in seconds before service is updated
        :param log_file_prefix: Log file prefix, default is "xcube-serve.log"
        :param log_to_stderr: Whether logging should be shown on stderr
        :return: service information dictionary
        """
        if config_file and cube_paths:
            raise ValueError("config_file and cube_paths cannot be given both")
        if config_file and styles:
            raise ValueError("config_file and styles cannot be given both")
        if config_file and aws_prof:
            raise ValueError("config_file and aws_profile cannot be given both")
        if config_file and aws_env:
            raise ValueError("config_file and aws_env cannot be given both")

        global SNAP_CPD_LIST
        if config_file:
            SNAP_CPD_LIST = _get_custom_color_list(config_file)

        log_dir = os.path.dirname(log_file_prefix)
        if log_dir and not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        options = tornado.options.options
        options.log_file_prefix = log_file_prefix or DEFAULT_LOG_PREFIX
        options.log_to_stderr = log_to_stderr
        enable_pretty_logging()

        tile_cache_capacity = parse_mem_size(tile_cache_size)

        config = None
        if cube_paths:
            config = new_default_config(cube_paths, styles, aws_prof=aws_prof, aws_env=aws_env)

        self.config_file = os.path.abspath(config_file) if config_file else None
        self.update_period = update_period
        self.update_timer = None
        self.config_error = None
        self.service_info = dict(port=port,
                                 address=address,
                                 started=datetime.now().isoformat(sep=' '),
                                 pid=os.getpid())

        self.context = ServiceContext(prefix=prefix,
                                      config=config,
                                      base_dir=base_dir,
                                      trace_perf=trace_perf,
                                      tile_comp_mode=tile_comp_mode,
                                      tile_cache_capacity=tile_cache_capacity)
        self._maybe_load_config()

        application.service_context = self.context
        application.time_of_last_activity = time.process_time()
        self.application = application

        # Register handlers for common termination signals
        signal.signal(signal.SIGINT, self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)

        self.server = application.listen(port, address=address or 'localhost')
        # Ensure we have the same event loop in all threads
        asyncio.set_event_loop_policy(_GlobalEventLoopPolicy(asyncio.get_event_loop()))
        self._maybe_load_config()
        self._maybe_install_update_check()
        self._shutdown_requested = False

    def start(self):
        address = self.service_info['address']
        port = self.service_info['port']
        test_url = self.context.get_service_url(f"http://{address}:{port}", "datasets")
        LOG.info(f'service running, listening on {address}:{port}, try {test_url}')
        LOG.info(f'press CTRL+C to stop service')
        if not self.context.config.get('Datasets', []) \
                and not self.context.config.get('DataStores', []):
            LOG.warning('no datasets or data stores configured')
        tornado.ioloop.PeriodicCallback(self._try_shutdown, 100).start()
        IOLoop.current().start()

    def stop(self, kill=False):
        """
        Stops the Tornado web server.
        """
        if kill:
            sys.exit(0)
        else:
            IOLoop.current().add_callback(self._on_shutdown)

    def _on_shutdown(self):

        LOG.info('stopping service...')

        # noinspection PyUnresolvedReferences,PyBroadException
        try:
            self.update_timer.cancel()
        except Exception:
            pass

        if self.server:
            self.server.stop()
            self.server = None

        IOLoop.current().stop()
        LOG.info('service stopped.')

    def _try_shutdown(self):
        if self._shutdown_requested:
            self._on_shutdown()

    # noinspection PyUnusedLocal
    def _sig_handler(self, sig, frame):
        LOG.warning(f'caught signal {sig}')
        self._shutdown_requested = True

    def _maybe_install_update_check(self):
        if self.config_file is None or self.update_period is None or self.update_period <= 0:
            return
        IOLoop.current().call_later(self.update_period, self._maybe_check_for_updates)

    def _maybe_check_for_updates(self):
        self._maybe_load_config()
        self._maybe_install_update_check()

    def _maybe_load_config(self):
        config_file = self.config_file
        if config_file is None:
            return

        try:
            stat = os.stat(config_file)
        except OSError as e:
            if self.config_error is None:
                LOG.error(f'configuration file {config_file!r}: {e}')
                self.config_error = e
            return

        if self.context.config_mtime != stat.st_mtime:
            self.context.config_mtime = stat.st_mtime
            try:
                self.context.config = load_json_or_yaml_config(config_file)
                self.config_error = None
                LOG.info(f'configuration file {config_file!r} successfully loaded')
            except ValueError as e:
                if self.config_error is None:
                    LOG.error(f'configuration file {config_file!r}: {e}')
                    self.config_error = e


# noinspection PyAbstractClass
class ServiceRequestHandler(RequestHandler):

    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)
        self._params = ServiceRequestParams(self)

    def set_caseless_query_arguments(self):
        self.request.query_arguments = caseless_dict(self.request.query_arguments or {})

    @property
    def service_context(self) -> ServiceContext:
        # noinspection PyUnresolvedReferences
        return self.application.service_context

    @property
    def base_url(self):
        return self.request.protocol + '://' + self.request.host

    @property
    def params(self) -> 'ServiceRequestParams':
        return self._params

    def set_default_headers(self):
        self.set_header('Server', f'xcube-server/{version}')
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods',
                        'GET,PUT,DELETE,OPTIONS')
        self.set_header('Access-Control-Allow-Headers',
                        'x-requested-with,access-control-allow-origin,'
                        'authorization,content-type')

    # noinspection PyUnusedLocal
    def options(self, *args, **kwargs):
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

    def get_query_arguments(self) -> Mapping[str, str]:
        handler = self.handler
        request = handler.request
        return {name: handler.get_query_argument(name) for name in request.query_arguments.keys()}

    def get_query_argument(self, name: str, default: Optional[str] = UNDEFINED) -> Optional[str]:
        """
        Get query argument.
        :param name: Query argument name
        :param default: Default value.
        :return: the value or none
        :raise: ServiceBadRequestError
        """
        value = self.handler.get_query_argument(name, default=default)
        if value == UNDEFINED:
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
    name_pattern = r'(?P<%s>[^\;\/\?\:\@\&\=\+\$\,]+)'
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


def new_default_config(cube_paths: List[str],
                       styles: Dict[str, Tuple] = None,
                       aws_prof: str = None,
                       aws_env: bool = False):
    aws_access_key_id = None
    aws_secret_access_key = None
    if aws_prof:
        aws_credentials_config = configparser.ConfigParser()
        aws_credentials_config.read(os.path.expanduser(os.path.join('~', '.aws', 'credentials')))
        aws_access_key_id = aws_credentials_config.get(aws_prof, 'aws_access_key_id', fallback=None)
        aws_secret_access_key = aws_credentials_config.get(aws_prof, 'aws_secret_access_key', fallback=None)
        if aws_access_key_id is None:
            raise ValueError(f'missing aws_access_key_id in AWS credentials')
        if aws_secret_access_key is None:
            raise ValueError(f'missing aws_secret_access_key in AWS credentials')
    if aws_env:
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID', aws_access_key_id)
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY', aws_secret_access_key)
        if aws_access_key_id is None:
            raise ValueError(f'environment variable AWS_ACCESS_KEY_ID not set')
        if aws_secret_access_key is None:
            raise ValueError(f'environment variable AWS_SECRET_ACCESS_KEY not set')

    dataset_configs = list()
    index = 0
    for cube_path in cube_paths:
        dataset_config = dict(Identifier=f"dataset_{index + 1}",
                              Format=guess_ml_dataset_format(cube_path),
                              Path=cube_path)
        if is_s3_url(cube_path):
            dataset_config.update(Title=cube_path.split('/')[-1],
                                  FileSystem='s3')
            if aws_access_key_id and aws_secret_access_key:
                dataset_config.update(AccessKeyId=aws_access_key_id,
                                      SecretAccessKey=aws_secret_access_key)
        else:
            dataset_config.update(Title=os.path.split(cube_path)[-1],
                                  FileSystem='file')
        dataset_configs.append(dataset_config)
        index += 1

    config = dict(Datasets=dataset_configs)
    if styles:
        color_mappings = {}
        for var_name, style_data in styles.items():
            try:
                value_min, value_max, color_bar_name = style_data
                style = dict(ValueRange=[value_min, value_max], ColorBar=color_bar_name)
            except (TypeError, ValueError):
                try:
                    value_min, value_max = style_data
                    style = dict(ValueRange=[value_min, value_max])
                except (TypeError, ValueError):
                    raise ValueError(f"illegal style: {var_name}={style_data!r}")
            color_mappings[var_name] = style
        config["Styles"] = [dict(Identifier="default", ColorMappings=color_mappings)]
    return config


# TODO (forman): fix this hack
def _get_custom_color_list(config_file):
    # global: too bad :(
    global SNAP_CPD_LIST
    config = load_configs(config_file) if config_file else {}
    styles = config.get('Styles')
    if isinstance(styles, list):
        for style in styles:
            cm = style.get('ColorMappings')
            if isinstance(cm, dict):
                for key in cm.keys():
                    if 'ColorFile' in cm[key]:
                        cf = cm[key]['ColorFile']
                        if cf not in SNAP_CPD_LIST:
                            SNAP_CPD_LIST.append(cf)
    return SNAP_CPD_LIST
