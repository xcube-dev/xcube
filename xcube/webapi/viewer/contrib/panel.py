# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from chartlets import Contribution


class Panel(Contribution):
    """Panel contribution.

    A panel is a UI-contribution to xcube Viewer.
    To become effective, instances of this class must be added
    to a ``chartlets.Extension`` instance exported from your extension
    module.

    Args:
        name: A name that is unique within the extension.
        title: An initial title for the panel.
    """

    def __init__(self, name: str, title: str | None = None):
        super().__init__(name, visible=False, title=title)
