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

import unittest
from typing import Sequence

from xcube.server.impl.framework.tornado import TornadoFramework
from ...mocks import mock_server


class TornadoFrameworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.application = MockApplication()
        self.io_loop = MockIOLoop()
        # noinspection PyTypeChecker
        self.framework = TornadoFramework(application=self.application,
                                          io_loop=self.io_loop)

    def test_start_and_update_and_stop(self):
        server = mock_server(framework=self.framework, config={})
        self.assertEqual(False, self.application.listening)
        self.assertEqual(False, self.io_loop.started)
        self.assertEqual(False, self.io_loop.stopped)
        self.framework.start(server.root_ctx)
        self.assertEqual(True, self.application.listening)
        self.assertEqual(True, self.io_loop.started)
        self.assertEqual(False, self.io_loop.stopped)
        self.framework.update(server.root_ctx)
        self.assertEqual(True, self.application.listening)
        self.assertEqual(True, self.io_loop.started)
        self.assertEqual(False, self.io_loop.stopped)
        self.framework.stop(server.root_ctx)
        self.assertEqual(True, self.application.listening)
        self.assertEqual(True, self.io_loop.started)
        self.assertEqual(True, self.io_loop.stopped)


class MockIOLoop:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class MockApplication:
    def __init__(self):
        self.handlers = []
        self.listening = False

    def add_handlers(self, domain: str, handlers: Sequence):
        self.handlers.extend(handlers)

    def listen(self, *args, **kwargs):
        self.listening = True
