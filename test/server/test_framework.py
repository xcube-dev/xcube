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
