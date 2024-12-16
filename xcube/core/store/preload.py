# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from enum import Enum
from typing import Callable


class PreloadState(Enum):
    """Preload process state."""

    created = "created"
    started = "started"
    stopped = "stopped"
    cancelled = "cancelled"
    failed = "failed"


class PreloadEventType(Enum):
    """Type of preload process event."""

    state = "state"
    progress = "progress"
    info = "info"
    warning = "warning"
    error = "error"


class PreloadEvent:
    """Event to occur during the preload process."""

    @classmethod
    def state(cls, state: PreloadState):
        """Create an event of type ``state``."""
        return PreloadEvent(PreloadEventType.state, state=state)

    @classmethod
    def progress(cls, progress: float):
        """Create an event of type ``process``."""
        return PreloadEvent(PreloadEventType.progress, progress=progress)

    @classmethod
    def info(cls, message: str):
        """Create an event of type ``info``."""
        return PreloadEvent(PreloadEventType.info, message=message)

    @classmethod
    def warning(cls, message: str, warning: Warning | None = None):
        """Create an event of type ``warning``."""
        return PreloadEvent(PreloadEventType.warning, message=message, warning=warning)

    @classmethod
    def error(cls, message: str, exception: Exception | None = None):
        """Create an event of type ``error``."""
        return PreloadEvent(
            PreloadEventType.error, message=message, exception=exception
        )

    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        type: PreloadEventType,
        state: PreloadState | None = None,
        progress: float | None = None,
        message: str | None = None,
        warning: Warning | None = None,
        exception: Exception | None = None,
    ):
        self.type = type
        self.state = state
        self.progress = progress
        self.message = message
        self.warning = warning
        self.exception = exception


class PreloadMonitor:

    def __init__(
        self,
        on_event: Callable[["PreloadMonitor", PreloadEvent], None] | None = None,
        on_done: Callable[["PreloadMonitor"], None] | None = None,
    ):
        self._is_cancelled = False
        if on_event:
            self.on_event = on_event
        if on_done:
            self.on_done = on_done

    @property
    def is_cancelled(self):
        """Is cancellation requested?"""
        return self._is_cancelled

    def cancel(self):
        """Request cancellation."""
        self._is_cancelled = True

    def on_event(self, event: PreloadEvent):
        """Called when an event occurs."""

    def on_done(self):
        """Called when the preload process is done and
        the data is ready to be accessed.

        The method is not called only on success.
        """


class PreloadHandle:
    """Represents an ongoing preload process."""

    def close(self):
        """Closes the preload.

        Should be called if the preloaded data is no longer needed.

        This method usually cleans the cache associated with
        this preload object.

        The default implementation does nothing.
        """

    def __enter__(self) -> "PreloadHandle":
        """Enter the context.

        Returns:
            This object.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context. Calls ``close()``."""
        self.close()
