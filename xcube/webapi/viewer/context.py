# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from functools import cached_property
from typing import Optional
from collections.abc import Mapping

import fsspec

from xcube.webapi.common.context import ResourcesContext


class ViewerContext(ResourcesContext):
    """Context for xcube Viewer API."""

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
