# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import threading
import time
import traceback
from abc import ABC
from collections.abc import Sequence
from typing import Any, List, Optional, Tuple, Type

import dask.callbacks
import dask.diagnostics

from xcube.util.assertions import assert_given, assert_true


class ProgressState:
    """Represents the state of progress."""

    def __init__(self, label: str, total_work: float, super_work: float):
        self._label = label
        self._total_work = total_work
        self._super_work = super_work
        self._super_work_ahead = 1.0
        self._exc_info = None
        self._traceback = None
        self._completed_work = 0.0
        self._finished = False
        self._start_time = None
        self._start_time = time.perf_counter()
        self._total_time = None

    @property
    def label(self) -> str:
        return self._label

    @property
    def total_work(self) -> float:
        return self._total_work

    @property
    def super_work(self) -> float:
        return self._super_work

    @property
    def completed_work(self) -> float:
        return self._completed_work

    @property
    def progress(self) -> float:
        return self._completed_work / self._total_work

    def to_super_work(self, work: float) -> float:
        return self._super_work * work / self._total_work

    @property
    def exc_info(self) -> Optional[tuple[type, BaseException, Any]]:
        return self._exc_info

    @exc_info.setter
    def exc_info(self, exc_info: tuple[type, BaseException, Any]):
        self._exc_info = exc_info

    @property
    def exc_info_text(self) -> Optional[tuple[str, str, list[str]]]:
        if not self.exc_info:
            return None
        exc_type, exc_value, exc_traceback = self.exc_info
        return (
            f"{type(exc_value).__name__}",
            f"{exc_value}",
            traceback.format_exception(exc_type, exc_value, exc_traceback),
        )

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def total_time(self) -> Optional[float]:
        return self._total_time

    @property
    def super_work_ahead(self) -> float:
        return self._super_work_ahead

    @super_work_ahead.setter
    def super_work_ahead(self, work: float):
        assert_true(work > 0, "work must be greater than zero")
        self._super_work_ahead = work

    def inc_work(self, work: float):
        assert_true(work > 0, "work must be greater than zero")
        self._completed_work += work

    def finish(self):
        self._finished = True
        self._total_time = time.perf_counter() - self._start_time


class ProgressObserver(ABC):
    """A progress observer is notified about nested state changes when using the
    :class:`observe_progress` context manager.
    """

    def on_begin(self, state_stack: Sequence[ProgressState]):
        """Called, if an observed code block begins execution."""

    def on_update(self, state_stack: Sequence[ProgressState]):
        """Called, if the progress state has changed within an observed code block."""

    def on_end(self, state_stack: Sequence[ProgressState]):
        """Called, if an observed block of code ends execution."""

    def activate(self):
        _ProgressContext.instance().add_observer(self)

    def deactivate(self):
        _ProgressContext.instance().remove_observer(self)


class _ProgressContext:
    _instance = None

    def __init__(self, *observers: ProgressObserver):
        self._observers = set(observers)
        self._state_stack = list()

    def add_observer(self, observer: ProgressObserver):
        self._observers.add(observer)

    def remove_observer(self, observer: ProgressObserver):
        self._observers.discard(observer)

    def emit_begin(self):
        for observer in self._observers:
            observer.on_begin(self._state_stack)

    def emit_update(self):
        for observer in self._observers:
            observer.on_update(self._state_stack)

    def emit_end(self):
        for observer in self._observers:
            observer.on_end(self._state_stack)

    def begin(self, label: str, total_work: float) -> ProgressState:
        super_work = self._state_stack[-1].super_work_ahead if self._state_stack else 1
        progress_state = ProgressState(label, total_work, super_work)
        self._state_stack.append(progress_state)
        self.emit_begin()
        return progress_state

    def end(self, exc_type, exc_value, exc_traceback) -> ProgressState:
        exc_info = tuple((exc_type, exc_value, exc_traceback))
        progress_state = self._state_stack[-1]
        progress_state.exc_info = exc_info if any(exc_info) else None
        progress_state.finish()
        self.emit_end()
        self._state_stack.pop()
        if self._state_stack:
            self._state_stack[-1].super_work_ahead = 1
        return progress_state

    def worked(self, work: float):
        assert_true(
            self._state_stack, "worked() method call is missing a current context"
        )
        assert_true(work > 0, "work must be greater than zero")
        for s in reversed(self._state_stack):
            s.inc_work(work)
            work = s.to_super_work(work)
        self.emit_update()

    def will_work(self, work: float):
        assert_true(
            self._state_stack, "will_work() method call is missing a current context"
        )
        # noinspection PyProtectedMember
        self._state_stack[-1].super_work_ahead = work

    @classmethod
    def instance(cls) -> "_ProgressContext":
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "_ProgressContext" = None) -> "_ProgressContext":
        cls._instance, old_instance = (instance or _ProgressContext()), cls._instance
        return old_instance


_ProgressContext.set_instance()


class new_progress_observers:
    """Takes zero or more progress observers and activates them in the enclosed context.
    Progress observers from an outer context will no longer be active.

    Args:
        observers: progress observers that will temporarily replace
            existing ones.
    """

    def __init__(self, *observers: ProgressObserver):
        self._observers = observers
        self._old_context = None

    def __enter__(self):
        self._old_context = _ProgressContext.set_instance(
            _ProgressContext(*self._observers)
        )

    def __exit__(self, type, value, traceback):
        _ProgressContext.set_instance(self._old_context)


class add_progress_observers:
    """Takes zero or more progress observers and uses them only in the
    enclosed context. Any progress observers from an outer context
    remain active.

    Args:
        observers: progress observers to be added temporarily.
    """

    def __init__(self, *observers: ProgressObserver):
        self._observers = observers

    def __enter__(self):
        for observer in self._observers:
            observer.activate()

    def __exit__(self, type, value, traceback):
        for observer in self._observers:
            observer.deactivate()


class observe_progress:
    """Context manager for observing progress in the enclosed context.

    Args:
        label: A label.
        total_work: The total work.
    """

    def __init__(self, label: str, total_work: float):
        assert_given(label, "label")
        assert_true(total_work > 0, "total_work must be greater than zero")
        self._label = label
        self._total_work = total_work
        self._state: Optional[ProgressState] = None

    @property
    def label(self) -> str:
        return self._label

    @property
    def total_work(self) -> float:
        return self._total_work

    @property
    def state(self) -> ProgressState:
        self._assert_used_correctly()
        return self._state

    def __enter__(self) -> "observe_progress":
        self._state = _ProgressContext.instance().begin(self._label, self._total_work)
        return self

    def __exit__(self, type, value, traceback):
        _ProgressContext.instance().end(type, value, traceback)

    # noinspection PyMethodMayBeStatic
    def worked(self, work: float):
        self._assert_used_correctly()
        _ProgressContext.instance().worked(work)

    # noinspection PyMethodMayBeStatic
    def will_work(self, work: float):
        self._assert_used_correctly()
        _ProgressContext.instance().will_work(work)

    def _assert_used_correctly(self):
        assert_true(
            self._state is not None,
            'observe_progress() must be used with "with" statement',
        )


class observe_dask_progress(dask.callbacks.Callback):
    """Observe progress made by Dask tasks.

    Args:
        label: A label.
        total_work: The total work.
        interval: Time in seconds to between progress reports.
        initial_interval: Time in seconds to wait before progress is
            reported.
    """

    def __init__(
        self,
        label: str,
        total_work: float,
        interval: float = 0.1,
        initial_interval: float = 0,
    ):
        super().__init__()
        assert_given(label, "label")
        assert_true(total_work > 0, "total_work must be greater than zero")
        self._label = label
        self._total_work = total_work
        self._state: Optional[ProgressState] = None
        self._initial_interval = initial_interval
        self._interval = interval
        self._last_worked = 0
        self._running = False

    def __enter__(self) -> "observe_dask_progress":
        super().__enter__()
        self._state = _ProgressContext.instance().begin(self._label, self._total_work)
        return self

    def __exit__(self, type, value, traceback):
        self._stop_thread()
        _ProgressContext.instance().end(type, value, traceback)
        super().__exit__(type, value, traceback)

    # noinspection PyUnusedLocal
    def _start(self, dsk):
        """Dask callback implementation."""
        self._dask_state = None
        self._start_time = time.perf_counter()
        # Start background thread
        self._running = True
        self._timer = threading.Thread(target=self._timer_func)
        self._timer.daemon = True
        self._timer.start()

    # noinspection PyUnusedLocal
    def _pretask(self, key, dsk, state):
        """Dask callback implementation."""
        self._dask_state = state

    # noinspection PyUnusedLocal
    def _posttask(self, key, result, dsk, state, worker_id):
        """Dask callback implementation."""
        self._update()

    # noinspection PyUnusedLocal
    def _finish(self, dsk, state, errored):
        """Dask callback implementation."""
        self._stop_thread()
        elapsed = time.perf_counter() - self._start_time
        if elapsed > self._initial_interval:
            self._update()

    def _timer_func(self):
        """Background thread for updating"""
        while self._running:
            elapsed = time.perf_counter() - self._start_time
            if elapsed > self._initial_interval:
                self._update()
            time.sleep(self._interval)

    def _update(self):
        dask_state = self._dask_state
        if not dask_state:
            return
        num_done = len(dask_state["finished"])
        num_tasks = num_done + sum(
            len(dask_state[k]) for k in ["ready", "waiting", "running"]
        )
        if num_done < num_tasks:
            work_fraction = num_done / num_tasks if num_tasks > 0 else 0
            worked = work_fraction * self._total_work
            work = worked - self._last_worked
            if work > 0:
                _ProgressContext.instance().worked(work)
                self._last_worked = worked

    def _stop_thread(self):
        if self._running:
            self._running = False
            self._timer.join()
