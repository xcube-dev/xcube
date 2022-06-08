# The MIT License (MIT)
# Copyright (c) 2021/2022 by the xcube team and contributors
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

import asyncio
import pkgutil
import unittest
import urllib3
import socket
import threading
import yaml

from contextlib import closing

from xcube.server.server import Server
from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.util import extension
from xcube.util.extension import ExtensionRegistry

from tornado.platform.asyncio import AnyThreadEventLoopPolicy
from xcube.server.impl.framework.tornado import TornadoFramework


# taken from https://stackoverflow.com/a/45690594
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class BaseTest(unittest.TestCase):
    port = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.port = find_free_port()
        data = pkgutil.get_data('test', 'test_config.yml')
        config = yaml.safe_load(data)
        config['port'] = cls.port
        config['address'] = 'localhost'
        asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        er = ExtensionRegistry()
        extension_component = config['extension_component']
        extension_name = config['extension_name']
        if extension_component:
            er.add_extension(loader=extension.import_component(
                extension_component),
                             point=EXTENSION_POINT_SERVER_APIS,
                             name=extension_name)
        server = Server(framework=TornadoFramework(), config=config,
                        extension_registry=er)
        tornado = threading.Thread(target=server.start)
        tornado.daemon = True
        tornado.start()

        cls.http = urllib3.PoolManager()

    def test_demonstrate_usage(self):
        url = f'http://localhost:{self.port}/I_do_not_exist'
        response = self.http.request('GET', url)
        self.assertEqual(404, response.status)
