# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

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
        extension_registry = mock_extension_registry(
            [
                ("datasets", dict(create_ctx=MockApiContext)),
            ]
        )

        config = {}
        with open(self.TEST_CONFIG, "w") as fp:
            yaml.safe_dump(config, fp)

        framework = MockFramework()
        server = Server(framework, config, extension_registry=extension_registry)

        server_ctx = server.ctx

        observer = ConfigChangeObserver(server, [self.TEST_CONFIG], 0.1)

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
