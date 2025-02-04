#  Copyright (c) 2018-2025 by xcube team and contributors
#  Permissions are hereby granted under the terms of the MIT License:
#  https://opensource.org/licenses/MIT.

import collections.abc
import unittest
from collections.abc import Mapping, MutableMapping
from test.s3test import s3_test
from test.webapi.helpers import get_api_ctx, get_server_config
from typing import Any, Optional, Union

import fsspec
from chartlets import ExtensionContext

from xcube.webapi.viewer.context import ViewerContext
from xcube.webapi.viewer.contrib import Panel


def get_viewer_ctx(
    server_config: Optional[Union[str, Mapping[str, Any]]] = None,
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

    def test_without_persistence(self):
        ctx = get_viewer_ctx()
        self.assertIsNone(ctx.persistence)

    def test_with_persistence(self):
        ctx = get_viewer_ctx("config-persistence.yml")
        self.assertIsInstance(ctx.persistence, MutableMapping)

    def test_panels_local(self):
        ctx = get_viewer_ctx("config-panels.yml")
        self.assert_extensions_ok(ctx.ext_ctx)

    @s3_test()
    def test_panels_s3(self, endpoint_url: str):
        server_config = get_server_config("config-panels.yml")
        bucket_name = "xcube-testing"
        base_dir = server_config["base_dir"]
        ext_path = server_config["Viewer"]["Augmentation"]["Path"]
        # Copy test extension to S3 bucket
        s3_fs: fsspec.AbstractFileSystem = fsspec.filesystem(
            "s3", endpoint_url=endpoint_url
        )
        s3_fs.put(f"{base_dir}/{ext_path}", f"{bucket_name}/{ext_path}", recursive=True)
        server_config["base_dir"] = f"s3://{bucket_name}"
        ctx = get_viewer_ctx(server_config)
        self.assert_extensions_ok(ctx.ext_ctx)

    def assert_extensions_ok(self, ext_ctx: ExtensionContext | None):
        self.assertIsNotNone(ext_ctx)
        self.assertIsInstance(ext_ctx.extensions, list)
        self.assertEqual(1, len(ext_ctx.extensions))
        self.assertIsInstance(ext_ctx.contributions, dict)
        self.assertEqual(1, len(ext_ctx.contributions))
        self.assertIn("panels", ext_ctx.contributions)
        self.assertIsInstance(ext_ctx.contributions["panels"], list)
        self.assertEqual(2, len(ext_ctx.contributions["panels"]))
        self.assertIsInstance(ext_ctx.contributions["panels"][0], Panel)
        self.assertIsInstance(ext_ctx.contributions["panels"][1], Panel)
