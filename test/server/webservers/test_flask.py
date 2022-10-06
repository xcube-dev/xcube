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

from xcube.server.webservers.flask import FlaskFramework


class TornadoFrameworkTest(unittest.TestCase):

    def setUp(self) -> None:
        self.framework = FlaskFramework()

    def test_config_schema(self):
        self.assertEqual(None, self.framework.config_schema)

    def test_add_static_routes(self):
        with self.assertRaises(NotImplementedError):
            self.framework.add_static_routes([], '')

    def test_add_routes(self):
        with self.assertRaises(NotImplementedError):
            self.framework.add_routes([], '')

    def test_start(self):
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            self.framework.start(None)

    def test_stop(self):
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            self.framework.stop(None)

    def test_update(self):
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            self.framework.update(None)

    def test_call_later(self):
        def my_func():
            pass

        with self.assertRaises(NotImplementedError):
            self.framework.call_later(0.1, my_func)

    def test_run_in_executor(self):
        def my_func():
            pass

        with self.assertRaises(NotImplementedError):
            self.framework.run_in_executor(None, my_func)
