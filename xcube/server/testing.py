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
import socket
import threading
import unittest
from contextlib import closing
from typing import Dict

import urllib3
from tornado.platform.asyncio import AnyThreadEventLoopPolicy

from xcube.server.impl.framework.tornado import TornadoFramework
from xcube.server.server import Server
from xcube.util.extension import ExtensionRegistry


# taken from https://stackoverflow.com/a/45690594
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class ServerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.port = find_free_port()
        config = {
            'port': self.port,
            'address': 'localhost'
        }
        self.add_config(config)
        asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        er = ExtensionRegistry()
        self.add_extension(er)
        self.server = Server(framework=TornadoFramework(), config=config,
                             extension_registry=er)
        tornado = threading.Thread(target=self.server.start)
        tornado.daemon = True
        tornado.start()

        self.http = urllib3.PoolManager()

    def tearDown(self) -> None:
        self.server.stop()

    def add_extension(self, er: ExtensionRegistry) -> None:
        pass

    def add_config(self, config: Dict) -> None:
        pass
