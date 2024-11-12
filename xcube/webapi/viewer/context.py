# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Optional
from collections.abc import Mapping
import sys

from chartlets import Extension
from chartlets import ExtensionContext
import fsspec

from xcube.constants import LOG
from xcube.server.api import Context
from xcube.webapi.common.context import ResourcesContext
from xcube.webapi.viewer.contrib import Panel

Extension.add_contrib_point("panels", Panel)


class ViewerContext(ResourcesContext):
    """Context for xcube Viewer API."""

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self.ext_ctx: ExtensionContext | None = None

    def on_update(self, prev_context: Optional[Context]):
        super().on_update(prev_context)
        viewer_config: dict = self.config.get("Viewer")
        if viewer_config:
            augmentation: dict | None = viewer_config.get("Augmentation")
            if augmentation:
                extension_refs: list[str] = augmentation["Extensions"]
                path: Path | None = None
                if "Path" in augmentation:
                    path = Path(augmentation["Path"])
                    if not path.is_absolute():
                        path = Path(self.base_dir) / path
                with prepend_sys_path(path):
                    LOG.info(f"Loading viewer extension(s) {','.join(extension_refs)}")
                    self.ext_ctx = ExtensionContext.load(
                        self.server_ctx, extension_refs
                    )

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


@contextmanager
def prepend_sys_path(path: Path | str | None):
    prev_sys_path = None
    if path is not None:
        LOG.warning(f"Temporarily prepending '{path}' to sys.path")
        prev_sys_path = sys.path
        sys.path = [str(path)] + sys.path
    try:
        yield path is not None
    finally:
        if prev_sys_path:
            sys.path = prev_sys_path
            LOG.info(f"Restored sys.path")
