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
from typing import Optional, Dict, Mapping, Any, List, Union, Callable, \
    Sequence

from xcube.constants import EXTENSION_POINT_SERVER_APIS, LOG
from xcube.server.api import Api
from xcube.server.api import ApiContext
from xcube.server.api import ApiRoute
from xcube.server.config import BASE_SERVER_CONFIG_SCHEMA
from xcube.server.config import Config
from xcube.server.context import Context
from xcube.server.framework import ServerFramework
from xcube.util.extension import ExtensionRegistry
from xcube.util.extension import get_extension_registry
from xcube.util.jsonschema import JsonObjectSchema


# TODO:
#   - generate OpenAPI document and add default endpoint "/openapi"
#   - allow for JSON schema for requests and responses (openAPI)
#   - introduce change management (per API?)
#     - detect server config changes
#     - detect API context patches
#   - fix logging, log server activities
#   - aim at 100% test coverage

class Server:
    """
    A REST server extendable by API extensions.

    APIs are registered using the extension point "xcube.server.api".

    :param framework: The web server framework to be used
    :param config: The server configuration.
    :param extension_registry: Optional extension registry.
        Defaults to xcube's default extension registry.
    """

    def __init__(
            self,
            framework: ServerFramework,
            config: Config,
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
        ctx.update(None)
        self._set_ctx(ctx)

    def start(self):
        """Start this server."""
        LOG.info(f'Starting service...')
        for api in self._apis:
            api.start(self.ctx)
        self._framework.start(self.ctx)

    def stop(self):
        """Stop this server."""
        LOG.info(f'Stopping service...')
        self._framework.stop(self.ctx)
        for api in self._apis:
            api.stop(self.ctx)
        self._ctx.dispose()

    def update(self, config: Config):
        """Update this server with new configuration."""
        ctx = self._new_ctx(config)
        ctx.update(prev_ctx=self._ctx)
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
        self._framework.call_later(delay, callback, *args, **kwargs)

    # Used mainly for testing
    @property
    def config_schema(self) -> JsonObjectSchema:
        """The effective JSON schema for the server configuration."""
        return self._config_schema

    # Used mainly for testing
    @property
    def ctx(self) -> "ServerContext":
        """The root (server) context."""
        return self._ctx

    def _set_ctx(self, ctx: "ServerContext"):
        self._ctx = ctx
        self._framework.update(ctx)

    def _new_ctx(self, config: Config):
        return ServerContext(self._apis,
                             self._config_schema.from_instance(config))

    @classmethod
    def load_apis(
            cls,
            extension_registry: Optional[ExtensionRegistry] = None
    ) -> Sequence[Api]:
        extension_registry = extension_registry \
                             or get_extension_registry()

        # Collect all registered APIs
        apis = [
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
        return sorted(apis,
                      key=lambda api: api_ref_counts[api.name])

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
    A server context holds the current server configuration and
    current API context objects.

    A new server context is created for any new server configuration.

    :param apis: The ordered sequence of loaded server APIs.
    :param config: The current server configuration.
    """

    def __init__(self,
                 apis: Sequence[Api],
                 config: Config):
        self._apis = apis
        self._config = config
        self._api_contexts: Dict[str, ApiContext] = dict()

    # @property
    # def apis(self) -> Tuple[Api]:
    #     return tuple(self._apis)

    @property
    def config(self) -> Config:
        return self._config

    def get_api_ctx(self, api_name: str) -> Optional[ApiContext]:
        return self._api_contexts.get(api_name)

    def set_api_ctx(self, api_name: str, api_ctx: ApiContext):
        self._assert_api_ctx_type(api_ctx, api_name)
        self._api_contexts[api_name] = api_ctx
        setattr(self, api_name, api_ctx)

    def update(self, prev_ctx: Optional["ServerContext"]):
        if prev_ctx is None:
            LOG.info(f'Applying initial configuration...')
        else:
            LOG.info(f'Applying configuration changes...')
        for api in self._apis:
            prev_api_ctx: Optional[ApiContext] = None
            if prev_ctx is not None:
                prev_api_ctx = prev_ctx.get_api_ctx(
                    api.name
                )
            for dep_api_name in api.required_apis:
                dep_api_ctx = self.get_api_ctx(dep_api_name)
                assert dep_api_ctx is not None
            next_api_ctx: Optional[ApiContext] = api.create_ctx(self)
            if next_api_ctx is not None:
                self.set_api_ctx(api.name, next_api_ctx)
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
            raise TypeError(f'API {api_name!r}:'
                            f' context must be instance of'
                            f' {ApiContext.__name__}')
