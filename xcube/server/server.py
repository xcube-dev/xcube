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
import warnings
from typing import Optional, Any, Dict

import tornado.ioloop
import tornado.web

from xcube.constants import EXTENSION_POINT_SERVER_APIS, LOG
from xcube.server.api import ServerApi
from xcube.server.config import ServerConfig
from xcube.server.context import ServerContextImpl
from xcube.util.extension import ExtensionRegistry
from xcube.util.extension import get_extension_registry


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


class Server:

    def start(
            self,
            server_config: Dict[str, Any],
            extension_registry: Optional[ExtensionRegistry] = None
    ):

        extension_registry = extension_registry or get_extension_registry()
        handlers = []
        api_configs = {}
        for ext in extension_registry.find_extensions(EXTENSION_POINT_SERVER_APIS):
            api_name = ext.name

            server_api: ServerApi = ext.component()
            handlers.extend(server_api.get_handlers())

            api_config = None
            if api_name in server_config:
                api_config = server_config[api_name]
                if not isinstance(api_config, dict):
                    raise ValueError(f'invalid configuration for API {api_name!r}')

            api_config_class = server_api.get_config_class()
            if api_config_class is not None:
                api_config = api_config_class.from_dict(api_config)
            elif api_name in server_config:
                LOG.warn(f'fFound configuration for API {api_name!r},'
                         f' but no configuration class was given.'
                         f' Using it as is.')

            api_configs[api_name] = api_config

        server_config = ServerConfig.from_dict(server_config)
        for k, v in api_configs.items():
            setattr(server_config, k, v)

        application = tornado.web.Application(handlers)

        server_context = ServerContextImpl(server_config)
        application.__server_context = server_context

        application.listen(server_config.port, address=server_config.address)
        tornado.ioloop.IOLoop.current().start()
