#  Copyright (c) 2018-2025 by xcube team and contributors
#  Permissions are hereby granted under the terms of the MIT License:
#  https://opensource.org/licenses/MIT.

from dataclasses import dataclass

from chartlets import Component


@dataclass(frozen=True)
class Markdown(Component):
    """A div-element in which the given markdown text is rendered."""

    text: str | None = None
    """The markdown text."""
