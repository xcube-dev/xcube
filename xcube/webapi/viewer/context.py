# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import sys
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import fsspec
from chartlets import Extension, ExtensionContext

from xcube.constants import LOG
from xcube.server.api import Context
from xcube.util.fspath import is_local_fs
from xcube.util.temp import new_temp_dir
from xcube.webapi.common.context import ResourcesContext
from xcube.webapi.viewer.contrib import Panel

Extension.add_contrib_point("panels", Panel)


class ViewerContext(ResourcesContext):
    """Context for xcube Viewer API."""

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self.ext_ctx: ExtensionContext | None = None
        self.persistence: MutableMapping | None = None

    def on_update(self, prev_context: Optional[Context]):
        super().on_update(prev_context)
        viewer_config: dict = self.config.get("Viewer")
        if not viewer_config:
            return
        persistence: dict | None = viewer_config.get("Persistence")
        if persistence:
            path = self.get_config_path(persistence, "Persistence")
            storage_options = persistence.get("StorageOptions")
            self.set_persistence(path, storage_options)
        augmentation: dict | None = viewer_config.get("Augmentation")
        if augmentation:
            path = augmentation.get("Path")
            extension_refs = augmentation["Extensions"]
            self.set_extension_context(path, extension_refs)

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

    def set_persistence(self, path: str, storage_options: dict[str, Any] | None):
        fs_root: tuple[fsspec.AbstractFileSystem, str] = fsspec.core.url_to_fs(
            path, **(storage_options or {})
        )
        fs, root = fs_root
        self.persistence = fs.get_mapper(root, create=True, check=True)
        LOG.info(f"Viewer persistence established for path {path!r}")

    def set_extension_context(self, path: str | None, extension_refs: list[str]):
        module_path = self.base_dir
        if path:
            module_path = f"{module_path}/{path}"
        fs_root: tuple[fsspec.AbstractFileSystem, str] = fsspec.core.url_to_fs(
            module_path
        )
        fs, fs_path = fs_root
        if is_local_fs(fs):
            local_module_path = Path(module_path)
        else:
            temp_module_path = new_temp_dir("xcube-viewer-aux-")
            LOG.warning(f"Downloading {module_path!r} to {temp_module_path!r}")
            fs.get(fs_path + "/**/*", temp_module_path + "/", recursive=True)
            local_module_path = Path(temp_module_path)
        with prepend_sys_path(local_module_path):
            LOG.info(f"Loading viewer extension(s) {','.join(extension_refs)}")
            self.ext_ctx = ExtensionContext.load(self.server_ctx, extension_refs)


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
            LOG.info("Restored sys.path")
