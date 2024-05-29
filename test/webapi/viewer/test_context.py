# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import unittest
from typing import Optional, Union, Any
from collections.abc import Mapping

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
        self.assertEqual(path.replace("\\", "/"), ctx.config_path)

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
