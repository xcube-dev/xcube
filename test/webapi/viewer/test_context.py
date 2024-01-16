# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

import collections.abc
import unittest
from typing import Optional, Union, Mapping, Any

from test.webapi.helpers import get_api_ctx
from xcube.webapi.viewer.context import ViewerContext


def get_viewer_ctx(
        server_config: Optional[Union[str, Mapping[str, Any]]] = None
) -> ViewerContext:
    return get_api_ctx("viewer", ViewerContext, server_config)


class ViewerContextTest(unittest.TestCase):

    def test_config_items_ok(self):
        ctx = get_viewer_ctx()
        self.assertIsInstance(ctx.config_items, collections.abc.Mapping)

    def test_config_path_ok(self):
        ctx = get_viewer_ctx()
        path = f"{ctx.config['base_dir']}/viewer"
        self.assertEqual(path.replace('\\', '/'), ctx.config_path)

        config_path = "s3://xcube-viewer-app/bc/dev/viewer/"
        config = dict(ctx.config)
        config.update(dict(Viewer=dict(Configuration=dict(Path=config_path))))
        ctx2 = get_viewer_ctx(server_config=config)
        self.assertEqual(config_path, ctx2.config_path)

        config_path = "memory:xcube-viewer"
        config = dict(ctx.config)
        config.update(dict(Viewer=dict(Configuration=dict(Path=config_path))))
        ctx2 = get_viewer_ctx(server_config=config)
        self.assertEqual(config_path, ctx2.config_path)

        config_path = "/xcube-viewer-app/bc/dev/viewer"
        config = dict(ctx.config)
        config.update(dict(Viewer=dict(Configuration=dict(Path=config_path))))
        ctx2 = get_viewer_ctx(server_config=config)
        self.assertEqual(config_path, ctx2.config_path)
