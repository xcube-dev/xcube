# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


import unittest

from xcube.webapi.viewer.components import Markdown


class ViewerComponentsTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_markdown(self):
        markdown = Markdown(text="_Hello_ **world**!")
        self.assertEqual(
            {"type": "Markdown", "text": "_Hello_ **world**!"},
            markdown.to_dict(),
        )
