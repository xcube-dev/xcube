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

import yaml

from xcube.core.dsio import rimraf
from xcube.server.helpers import ConfigChangeObserver
from xcube.server.server import Server
from .test_server import MockApiContext
from .test_server import MockFramework
from .test_server import mock_extension_registry


class ConfigChangeObserverTest(unittest.TestCase):

    TEST_CONFIG = "config.yaml"

    def tearDown(self) -> None:
        rimraf(self.TEST_CONFIG)

    def test_config_change_observer(self):
        extension_registry = mock_extension_registry([
            ("datasets", dict(create_ctx=MockApiContext)),
        ])

        config = {}
        with open(self.TEST_CONFIG, "w") as fp:
            yaml.safe_dump(config, fp)

        framework = MockFramework()
        server = Server(
            framework, config,
            extension_registry=extension_registry
        )

        server_ctx = server.ctx

        observer = ConfigChangeObserver(server,
                                        [self.TEST_CONFIG],
                                        0.1)

        self.assertEqual(0, framework.call_later_count)
        self.assertIs(server_ctx, server.ctx)

        observer.check()
        self.assertEqual(1, framework.call_later_count)
        self.assertIs(server_ctx, server.ctx)

        observer.check()
        self.assertEqual(2, framework.call_later_count)
        self.assertIs(server_ctx, server.ctx)

        with open(self.TEST_CONFIG, "w") as fp:
            yaml.safe_dump(config, fp)

        observer.check()
        self.assertEqual(3, framework.call_later_count)
        self.assertIsNot(server_ctx, server.ctx)
