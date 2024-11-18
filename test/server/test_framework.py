# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.plugin import init_plugin
from xcube.server.framework import Framework
from xcube.server.framework import get_framework_class
from xcube.server.framework import get_framework_names
from xcube.util.plugin import get_extension_registry


class FrameworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        extension_registry = get_extension_registry()
        init_plugin(extension_registry)

    def test_get_framework_names(self):
        framework_names = get_framework_names()
        self.assertEqual({"tornado", "flask"}, set(framework_names))

    def test_get_framework_class(self):
        framework_class = get_framework_class("tornado")
        self.assertIsInstance(framework_class, type)
        self.assertTrue(issubclass(framework_class, Framework))
