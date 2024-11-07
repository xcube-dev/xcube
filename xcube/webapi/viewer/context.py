# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from functools import cached_property
from typing import Optional
from collections.abc import Mapping

from dashipy import ExtensionContext
import fsspec

from xcube.server.api import Context
from xcube.webapi.common.context import ResourcesContext


class ViewerContext(ResourcesContext):
    """Context for xcube Viewer API."""

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self.ext_ctx: ExtensionContext | None = None

    def on_update(self, prev_context: Optional[Context]):
        super().on_update(prev_context)
        if "Viewer" not in self.config:
            return None
        extension_infos: list[dict] = self.config["Viewer"].get("Extensions")
        if extension_infos:
            extension_modules = [e["Path"] for e in extension_infos if "Path" in e]
            print("----------------------->", extension_modules)
            extensions = ExtensionContext.load_extensions(extension_modules)
            self.ext_ctx = ExtensionContext(self.server_ctx, extensions)

    @cached_property
    def config_items(self) -> Optional[Mapping[str, bytes]]:
        if self.config_path is None:
            return None
        return fsspec.get_mapper(self.config_path)

    @cached_property
    def config_path(self) -> Optional[str]:
        if "Viewer" not in self.config:
            return None
        return self.get_config_path(
            self.config["Viewer"].get("Configuration", {}),
            "'Configuration' item of 'Viewer'",
        )
