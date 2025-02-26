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
        icon: Name of a Material Design icon, see https://fonts.google.com/icons.
        position: If given, place the panel at the given position.
        after: If given, place the panel after the given position or name.
        before: If given, place the panel before the given position or name.
    """

    def __init__(
        self,
        name: str,
        title: str | None = None,
        icon: str | None = None,
        position: int | None = None,
        after: int | str | None = None,
        before: int | str | None = None,
    ):
        super().__init__(
            name,
            visible=False,
            title=title,
            icon=icon,
            position=position,
            after=after,
            before=before,
        )
