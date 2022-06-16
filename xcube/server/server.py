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

import concurrent.futures
import copy
from typing import (Optional, Dict, Any, Union,
                    Callable, Sequence, Awaitable, Tuple, List, Type)

from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.constants import LOG
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_subclass
from xcube.util.extension import ExtensionRegistry
from xcube.util.extension import get_extension_registry
from xcube.util.jsonschema import JsonObjectSchema
from .api import Api
from .api import ApiContext
from .api import ApiContextT
from .api import ApiRoute
from .api import Context
from .api import ReturnT
from .api import ServerConfig
from .asyncexec import AsyncExecution
from .config import BASE_SERVER_CONFIG_SCHEMA
from .framework import Framework


class Server(AsyncExecution):
    """
    A REST server extendable by API extensions.

    APIs are registered using the extension point ".api".

    TODO:
      - introduce change management (per API?)
        - detect API context patches
      - address common server configuration
        - camel case vs snake case parameters names ("BaseDir" vs "base_dir")
        - first capital letter in parameter names ("Address" vs "address")
        - let server types decide on vocabulary (config aliases)?
        - configure server types by meta-configuration,
          e.g. server name, description, config aliases, APIs used.
      - use any given request JSON schema to validate requests
      - aim at 100% test coverage

    :param framework: The web server framework to be used
    :param config: The server configuration.
    :param extension_registry: Optional extension registry.
        Defaults to xcube's default extension registry.
    """

    def __init__(
            self,
            framework: Framework,
            config: ServerConfig,
            extension_registry: Optional[ExtensionRegistry] = None,
    ):
        apis = self.load_apis(extension_registry)
        for api in apis:
            LOG.info(f'Loaded service API {api.name!r}')
        handlers = self.collect_api_routes(apis)
        framework.add_routes(handlers)
        self._framework = framework
        self._apis = apis
        self._config_schema = self.get_effective_config_schema(apis)
        ctx = self._new_ctx(config)
        ctx.on_update(None)
        self._set_ctx(ctx)

    @property
    def framework(self) -> Framework:
        """The web server framework use by this server."""
        return self._framework

    @property
    def apis(self) -> Tuple[Api]:
        """The APIs supported by this server."""
        return self._apis

    @property
    def config_schema(self) -> JsonObjectSchema:
        """The effective JSON schema for the server configuration."""
        return self._config_schema

    @property
    def ctx(self) -> "ServerContext":
        """The current server context."""
        return self._ctx

    def _set_ctx(self, ctx: "ServerContext"):
        self._ctx = ctx
        self._framework.update(ctx)

    def _new_ctx(self, config: ServerConfig) -> "ServerContext":
        config = dict(config)
        for key in tuple(config.keys()):
            if key not in self._config_schema.properties:
                LOG.warning(f'Configuration setting {key!r} ignored,'
                            f' because there is no schema describing it.')
                config.pop(key)
        return ServerContext(self,
                             self._config_schema.from_instance(config))

    def start(self):
        """Start this server."""
        LOG.info(f'Starting service...')
        for api in self._apis:
            api.on_start(self.ctx)
        self._framework.start(self.ctx)

    def stop(self):
        """Stop this server."""
        LOG.info(f'Stopping service...')
        self._framework.stop(self.ctx)
        for api in self._apis:
            api.on_stop(self.ctx)
        self._ctx.on_dispose()

    def update(self, config: ServerConfig):
        """Update this server with new configuration."""
        ctx = self._new_ctx(config)
        ctx.on_update(prev_ctx=self._ctx)
        self._set_ctx(ctx)

    def call_later(self,
                   delay: Union[int, float],
                   callback: Callable,
                   *args,
                   **kwargs):
        """
        Executes the given callable *callback* after *delay* seconds.

        :param delay: Delay in seconds.
        :param callback: Callback to be called.
        :param args: Positional arguments passed to *callback*.
        :param kwargs: Keyword arguments passed to *callback*.
        """
        return self._framework.call_later(
            delay, callback, *args, **kwargs
        )

    def run_in_executor(
            self,
            executor: Optional[concurrent.futures.Executor],
            function: Callable[..., ReturnT],
            *args: Any,
            **kwargs: Any
    ) -> Awaitable[ReturnT]:
        """
        Concurrently runs a *function* in a ``concurrent.futures.Executor``.
        If *executor* is ``None``, the framework's default
        executor will be used.

        :param executor: An optional executor.
        :param function: The function to be run concurrently.
        :param args: Positional arguments passed to *function*.
        :param kwargs: Keyword arguments passed to *function*.
        :return: The awaitable return value of *function*.
        """
        return self._framework.run_in_executor(
            executor, function, *args, **kwargs
        )

    @classmethod
    def load_apis(
            cls,
            extension_registry: Optional[ExtensionRegistry] = None
    ) -> Tuple[Api]:
        extension_registry = extension_registry \
                             or get_extension_registry()

        # Collect all registered APIs
        apis: List[Api] = [
            ext.component
            for ext in extension_registry.find_extensions(
                EXTENSION_POINT_SERVER_APIS
            )
        ]

        api_lookup = {api.name: api for api in apis}

        def assert_required_apis_available():
            # Assert that required APIs are available.
            for api in apis:
                for req_api_name in api.required_apis:
                    if req_api_name not in api_lookup:
                        raise ValueError(f'API {api.name!r}: missing API'
                                         f' dependency {req_api_name!r}')

        assert_required_apis_available()

        def count_api_refs(api: Api) -> int:
            # Count the number of times the given API is referenced.
            dep_sum = 0
            for req_api_name in api.required_apis:
                dep_sum += count_api_refs(api_lookup[req_api_name]) + 1
            for opt_api_name in api.optional_apis:
                if opt_api_name in api_lookup:
                    dep_sum += count_api_refs(api_lookup[opt_api_name]) + 1
            return dep_sum

        # Count the number of times each API is referenced.
        api_ref_counts = {
            api.name: count_api_refs(api)
            for api in apis
        }

        # Return an ordered dict sorted by an API's reference count
        return tuple(sorted(apis,
                            key=lambda api: api_ref_counts[api.name]))

    @classmethod
    def collect_api_routes(cls, apis: Sequence[Api]) -> Sequence[ApiRoute]:
        handlers = []
        for api in apis:
            handlers.extend(api.routes)
        return handlers

    @classmethod
    def get_effective_config_schema(cls, apis: Sequence[Api]) \
            -> JsonObjectSchema:
        effective_config_schema = copy.deepcopy(BASE_SERVER_CONFIG_SCHEMA)
        for api in apis:
            api_config_schema = api.config_schema
            if api_config_schema is not None:
                assert isinstance(api_config_schema, JsonObjectSchema)
                for k, v in api_config_schema.properties.items():
                    if k in effective_config_schema.properties:
                        raise ValueError(f'API {api.name!r}:'
                                         f' configuration parameter {k!r}'
                                         f' is already defined.')
                    effective_config_schema.properties[k] = v
                if api_config_schema.required:
                    effective_config_schema.required.update(
                        api_config_schema.required
                    )
        return effective_config_schema


class ServerContext(Context):
    """
    The server context holds the current server configuration and
    the API context objects that depend on that specific configuration.

    A new server context is created for any new server configuration,
    which in turn will cause all API context objects to be updated.

    The constructor shall not be called directly.

    :param server: The server.
    :param config: The current server configuration.
    """

    def __init__(self,
                 server: Server,
                 config: ServerConfig):
        self._server = server
        self._config = config
        self._api_contexts: Dict[str, Context] = dict()

    @property
    def apis(self) -> Tuple[Api]:
        return self._server.apis

    @property
    def config(self) -> ServerConfig:
        return self._config

    def get_api_ctx(self,
                    api_name: str,
                    cls: Optional[Type[ApiContextT]] = None) \
            -> Optional[ApiContextT]:
        api_ctx = self._api_contexts.get(api_name)
        if cls is not None:
            assert_subclass(cls, ApiContext, name='cls')
            assert_instance(api_ctx, cls,
                            name=f'api_ctx (context of API {api_name!r})')
        return api_ctx

    def _set_api_ctx(self, api_name: str, api_ctx: ApiContext):
        assert_instance(api_ctx, ApiContext,
                        name=f'api_ctx (context of API {api_name!r})')
        self._api_contexts[api_name] = api_ctx
        setattr(self, api_name, api_ctx)

    def call_later(self,
                   delay: Union[int, float],
                   callback: Callable,
                   *args,
                   **kwargs) -> object:
        return self._server.call_later(delay, callback,
                                       *args, **kwargs)

    def run_in_executor(self,
                        executor: Optional[concurrent.futures.Executor],
                        function: Callable[..., ReturnT],
                        *args: Any,
                        **kwargs: Any) -> Awaitable[ReturnT]:
        return self._server.run_in_executor(executor, function,
                                            *args, **kwargs)

    def on_update(self, prev_ctx: Optional["ServerContext"]):
        if prev_ctx is None:
            LOG.info(f'Applying initial configuration...')
        else:
            LOG.info(f'Applying configuration changes...')
        for api in self.apis:
            prev_api_ctx: Optional[ApiContext] = None
            if prev_ctx is not None:
                prev_api_ctx = prev_ctx.get_api_ctx(api.name)
                assert prev_api_ctx is not None
            for dep_api_name in api.required_apis:
                dep_api_ctx = self.get_api_ctx(dep_api_name)
                assert dep_api_ctx is not None
            next_api_ctx: Optional[ApiContext] = api.create_ctx(self)
            self._set_api_ctx(api.name, next_api_ctx)
            next_api_ctx.on_update(prev_api_ctx)
            if prev_api_ctx is not None \
                    and prev_api_ctx is not next_api_ctx:
                prev_api_ctx.on_dispose()

    def on_dispose(self):
        for api_name in reversed([api.name for api in self.apis]):
            api_ctx = self.get_api_ctx(api_name)
            if api_ctx is not None:
                api_ctx.on_dispose()
