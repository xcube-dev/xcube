# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.server.webservers.flask import FlaskFramework


class TornadoFrameworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.framework = FlaskFramework()

    def test_config_schema(self):
        self.assertEqual(None, self.framework.config_schema)

    def test_add_static_routes(self):
        with self.assertRaises(NotImplementedError):
            self.framework.add_static_routes([], "")

    def test_add_routes(self):
        with self.assertRaises(NotImplementedError):
            self.framework.add_routes([], "")

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
