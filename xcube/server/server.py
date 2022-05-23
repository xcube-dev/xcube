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

import copy
import logging
from typing import Optional, Dict, Mapping, Any

import tornado.ioloop
import tornado.web

from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.server.api import Api
from xcube.server.api import ApiContext
from xcube.server.api import ApiHandler
from xcube.server.api import _SERVER_CONTEXT_ATTR_NAME
from xcube.server.config import BASE_SERVER_CONFIG_SCHEMA
from xcube.server.config import Config
from xcube.server.context import Context
from xcube.util.extension import ExtensionRegistry
from xcube.util.extension import get_extension_registry
from xcube.util.jsonschema import JsonObjectSchema


class Server:
    """
    A REST server extendable by API extensions.

    APIs are registered using the extension point "xcube.server.api".

    :param config: The server configuration.
    :param io_loop: Optional Tornado I/O loop.
        Defaults to Tornado's default I/O loop.
    :param extension_registry: Optional extension registry.
        Defaults to xcube's default extension registry.
    """

    def __init__(
            self,
            config: Config,
            io_loop: Optional[tornado.ioloop.IOLoop] = None,
            extension_registry: Optional[ExtensionRegistry] = None
    ):
        apis = self.load_apis(extension_registry)
        handlers = self.get_api_handlers(apis)
        self._apis = apis
        self._config_schema = self.get_effective_config_schema(apis)
        self._configure_tornado_logger()
        self._io_loop = io_loop or tornado.ioloop.IOLoop.current()
        self._application = tornado.web.Application(handlers)
        root_ctx = self._create_ctx(config)
        root_ctx.update(None)
        self._root_ctx = root_ctx

    def start(self):
        config = self._root_ctx.config
        port = config["port"]
        address = config["address"]
        self._application.listen(port, address=address)
        for api in self._apis.values():
            api.on_start(self._root_ctx, self._io_loop)
        self._io_loop.start()

    def stop(self):
        for api in self._apis.values():
            api.on_stop(self._root_ctx, self._io_loop)
        self._io_loop.stop()
        self._root_ctx.dispose()

    def update(self, config: Config):
        root_ctx = self._create_ctx(config)
        root_ctx.update(self._root_ctx)
        self._set_root_ctx(root_ctx)

    def _set_root_ctx(self, root_ctx: "ServerContext"):
        self._root_ctx = root_ctx
        # Register root context in Tornado application, so we
        # can access all contexts from the request handlers later.
        setattr(self._application, _SERVER_CONTEXT_ATTR_NAME, root_ctx)

    def _create_ctx(self, config: Config):
        return ServerContext(self._apis,
                             self._config_schema.from_instance(config))

    @classmethod
    def load_apis(
            cls,
            extension_registry: Optional[ExtensionRegistry] = None
    ) -> Dict[str, Api]:
        extension_registry = extension_registry \
                             or get_extension_registry()

        apis = {
            ext.name: ext.component
            for ext in extension_registry.find_extensions(
                EXTENSION_POINT_SERVER_APIS
            )
        }

        def count_api_deps(api: Api) -> int:
            dep_sum = 0
            for api_name in api.dependencies:
                dep_sum += count_api_deps(apis[api_name]) + 1
            return dep_sum

        api_dep_counts = {
            api.name: count_api_deps(api)
            for api in apis.values()
        }

        return {
            api.name: api
            for api in sorted(apis.values(),
                              key=lambda api: api_dep_counts[api.name])
        }

    @classmethod
    def get_api_handlers(cls, apis: Dict[str, Api]):
        handlers = []
        for api in apis.values():
            for route in api.routes:
                api_route_path = route[0]
                api_handler_cls = route[1]
                if not issubclass(api_handler_cls, ApiHandler):
                    raise TypeError(f'API {api.name!r}:'
                                    f' route {api_route_path}:'
                                    f' handler must be instance of'
                                    f' {ApiHandler.__name__}')
                api_handler_cls.api_name = api.name
            handlers.extend(api.routes)
        return handlers

    @classmethod
    def get_effective_config_schema(cls, apis: Dict[str, Api]):
        effective_config_schema = copy.deepcopy(BASE_SERVER_CONFIG_SCHEMA)
        for api_name, api in apis.items():
            api_config_schema = api.config_schema
            if api_config_schema is not None:
                if not isinstance(api_config_schema, JsonObjectSchema):
                    raise TypeError(f'API {api_name!r}:'
                                    f' configuration JSON schema'
                                    f' must be instance of'
                                    f' {JsonObjectSchema.__name__}')
                for k, v in api_config_schema.properties.items():
                    if k in effective_config_schema.properties:
                        raise ValueError(f'API {api_name!r}:'
                                         f' configuration parameter {k!r}'
                                         f' is already defined.')
                    effective_config_schema.properties[k] = v
                if api_config_schema.required:
                    effective_config_schema.required.update(
                        api_config_schema.required
                    )
        return effective_config_schema

    @staticmethod
    def _configure_tornado_logger():
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


class ServerContext(Context):
    """
    A server context holds the current server configuration and
    current API context objects.

    A new server context is created for any new server configuration.

    :param apis: The loaded server APIs.
    :param config: The current server configuration.
    """

    def __init__(self,
                 apis: Mapping[str, Api],
                 config: Config):
        self._apis = apis
        self._config = config
        self._api_contexts: Dict[str, ApiContext] = dict()

    @property
    def config(self) -> Config:
        assert self._config is not None
        return self._config

    def get_api_ctx(self, api_name: str) -> Optional[ApiContext]:
        return self._api_contexts.get(api_name)

    def set_api_ctx(self, api_name: str, api_ctx: ApiContext):
        self._assert_api_ctx_type(api_ctx, api_name)
        self._api_contexts[api_name] = api_ctx
        setattr(self, api_name, api_ctx)

    def update(self, prev_root_ctx: Optional["ServerContext"]):
        for api_name, api in self._apis.items():
            prev_api_ctx: Optional[ApiContext] = None
            if prev_root_ctx is not None:
                prev_api_ctx = prev_root_ctx.get_api_ctx(
                    api_name
                )
            next_api_ctx: Optional[ApiContext] = api.create_ctx(self)
            if next_api_ctx is not None:
                self.set_api_ctx(api_name, next_api_ctx)
                next_api_ctx.update(prev_api_ctx)
            elif prev_api_ctx is not None:
                # There is no next context so dispose() the previous one
                prev_api_ctx.dispose()

    def dispose(self):
        reversed_api_contexts = reversed(list(self._api_contexts.items()))
        for api_name, api_ctx in reversed_api_contexts:
            api_ctx.dispose()

    @classmethod
    def _assert_api_ctx_type(cls, api_ctx: Any, api_name: str):
        if not isinstance(api_ctx, ApiContext):
            raise TypeError(f'API {api_name}:'
                            f' context must be instance of'
                            f' {ApiContext.__name__}')
