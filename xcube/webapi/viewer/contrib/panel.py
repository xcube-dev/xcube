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
        icon: Name of a [Material Design icon](https://fonts.google.com/icons)
            to be used for the icon button representing the panel in the
            viewer's sidebar.
        position: If given, place the panel's icon button at the given position
            in the viewer's sidebar.
    """

    def __init__(
        self,
        name: str,
        title: str | None = None,
        icon: str | None = None,
        position: int | None = None,
    ):
        super().__init__(
            name,
            visible=False,
            title=title,
            icon=icon,
            position=position,
        )
