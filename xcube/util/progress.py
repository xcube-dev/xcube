# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import threading
import traceback
from abc import ABC
from time import sleep
from timeit import default_timer
from typing import Sequence, Optional, Any, Tuple, Type, List

from xcube.util.assertions import assert_condition, assert_given


class ProgressState:
    """Represents the state of progress."""

    def __init__(self, label: str, total_work: float, super_work: float):
        self._label = label
        self._total_work = total_work
        self._super_work = super_work
        self._super_work_ahead = 1.
        self._exc_info = None
        self._traceback = None
        self._completed_work = 0.
        self._finished = False

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
    def exc_info(self) -> Optional[Tuple[Type, BaseException, Any]]:
        return self._exc_info

    @exc_info.setter
    def exc_info(self, exc_info: Tuple[Type, BaseException, Any]):
        self._exc_info = exc_info

    @property
    def exc_info_text(self) -> Optional[Tuple[str, str, List[str]]]:
        if not self.exc_info:
            return None
        exc_type, exc_value, exc_traceback = self.exc_info
        return (f'{type(exc_value).__name__}',
                f'{exc_value}',
                traceback.format_exception(exc_type, exc_value, exc_traceback))

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def super_work_ahead(self) -> float:
        return self._super_work_ahead

    @super_work_ahead.setter
    def super_work_ahead(self, work: float):
        assert_condition(work > 0, 'work must be greater than zero')
        self._super_work_ahead = work

    def inc_work(self, work: float):
        assert_condition(work > 0, 'work must be greater than zero')
        self._completed_work += work

    def finish(self):
        self._finished = True


class ProgressObserver(ABC):
    """
    A progress observer is notified about nested state changes when using the
    :class:observe_progress context manager.
    """

    def on_begin(self, state_stack: Sequence[ProgressState]):
        """
        Called, if an observed code block begins execution.
        """

    def on_update(self, state_stack: Sequence[ProgressState]):
        """
        Called, if the progress state has changed within an observed code block.
        """

    def on_end(self, state_stack: Sequence[ProgressState]):
        """
        Called, if an observed block of code ends execution.
        """

    def activate(self):
        _ProgressContext.instance().add_observer(self)

    def deactivate(self):
        _ProgressContext.instance().remove_observer(self)


class ThreadedProgressObserver(ProgressObserver):
    """
    A threaded Progress observer adapted from Dask's ProgressBar class.
    """
    def __init__(self,
                 delegate: Any,
                 minimum: float = 0,
                 dt: float = 1):
        """

        :type dt: float
        :type minimum: float
        """
        super().__init__()
        assert_condition(dt >= 0, "The timer's time step must be >=0")
        assert_condition(minimum >= 0, "The timer's minimum must be >=0")
        self._delegate = delegate
        self._running = False
        self._start_time = None
        self._minimum = minimum
        self._dt = dt
        self._width = 100
        self._current_sender = None
        self._state_stack: [ProgressState] = []

    def _timer_func(self):
        """Background thread for updating the progress bar"""
        while self._running:
            elapsed = default_timer() - self._start_time
            if elapsed > self._minimum:
                self._update_state(elapsed)
            sleep(self._dt)

    def _start_timer(self):
        self._width = self._state_stack[0].total_work
        self._start_time = default_timer()
        # Start background thread
        self._running = True
        self._timer = threading.Thread(target=self._timer_func)
        self._timer.daemon = True
        self._timer.start()

    def _stop_timer(self, errored):
        self._running = False
        self._timer.join()
        elapsed = default_timer() - self._start_time
        self.last_duration = elapsed
        if elapsed < self._minimum:
            return
        if not errored:
            self._delegate(self._current_sender, elapsed, self._state_stack)
        else:
            self._update_state(elapsed)

    def _update_state(self, elapsed):
        if not self._state_stack:
            self._delegate.callback(self._current_sender, 0, elapsed)
            return
        state = self._state_stack[0]

        if state.completed_work < state.total_work:
            self._delegate.callback(self._current_sender, elapsed, self._state_stack)

    def on_begin(self, state_stack: Sequence[ProgressState]):
        self._delegate.on_begin(state_stack)

        assert_given(state_stack, name='state_stack')
        self._state_stack = state_stack
        self._current_sender = "on_begin"
        self._start_timer()

    def on_update(self, state_stack: Sequence[ProgressState]):
        self._delegate.on_update(state_stack)

        assert_given(state_stack, name="state_stack")
        self._state_stack = state_stack
        self._current_sender = "on_update"

    def on_end(self, state_stack: Sequence[ProgressState]):
        self._delegate.on_end(state_stack)

        assert_given(state_stack, name="state_stack")
        self._state_stack = state_stack
        self._current_sender = "on_end"
        self._stop_timer(False)

    def callback(self, sender: str, elapsed: float, state_stack: Sequence[ProgressState]):
        assert_given(state_stack, 'state_stack')

        if not hasattr(self._delegate,"callback"):
            raise ValueError("Please implement method callback.")
        self._delegate.callback(sender, elapsed, state_stack)


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

    def begin(self, label: str, total_work: float):
        super_work = self._state_stack[-1].super_work_ahead if self._state_stack else 1
        self._state_stack.append(ProgressState(label, total_work, super_work))
        self.emit_begin()

    def end(self, exc_type, exc_value, exc_traceback):
        exc_info = tuple((exc_type, exc_value, exc_traceback))
        self._state_stack[-1].exc_info = exc_info if any(exc_info) else None
        self._state_stack[-1].finish()
        self.emit_end()
        self._state_stack.pop()
        if self._state_stack:
            self._state_stack[-1].super_work_ahead = 1

    def worked(self, work: float):
        assert_condition(self._state_stack, 'worked() method call is missing a current context')
        assert_condition(work > 0, 'work must be greater than zero')
        for s in reversed(self._state_stack):
            s.inc_work(work)
            work = s.to_super_work(work)
        self.emit_update()

    def will_work(self, work: float):
        assert_condition(self._state_stack, 'will_work() method call is missing a current context')
        # noinspection PyProtectedMember
        self._state_stack[-1].super_work_ahead = work

    @classmethod
    def instance(cls) -> '_ProgressContext':
        return cls._instance

    @classmethod
    def set_instance(cls, instance: '_ProgressContext' = None) -> '_ProgressContext':
        cls._instance, old_instance = (instance or _ProgressContext()), cls._instance
        return old_instance


_ProgressContext.set_instance()


class new_progress_observers:
    """
    Takes zero or more progress observers and activates them in the enclosed context.
    Progress observers from an outer context will no longer be active.
    """

    def __init__(self, *observers: ProgressObserver):
        self._observers = observers
        self._old_context = None

    def __enter__(self):
        self._old_context = _ProgressContext.set_instance(_ProgressContext(*self._observers))

    def __exit__(self, type, value, traceback):
        _ProgressContext.set_instance(self._old_context)


class add_progress_observers:
    """
    Takes zero or more progress observers and uses them only in the enclosed context.
    Any progress observers from an outer context remain active.
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
    """
    Context manager for observing progress in the enclosed context.
    """

    def __init__(self, label: str, total_work: float):
        assert_given(label, 'label')
        assert_condition(total_work > 0, 'total_work must be greater than zero')
        self.label = label
        self.total_work = total_work

    def __enter__(self) -> 'observe_progress':
        _ProgressContext.instance().begin(self.label, self.total_work)
        return self

    def __exit__(self, type, value, traceback):
        _ProgressContext.instance().end(type, value, traceback)

    # noinspection PyMethodMayBeStatic
    def worked(self, work: float):
        _ProgressContext.instance().worked(work)

    # noinspection PyMethodMayBeStatic
    def will_work(self, work: float):
        _ProgressContext.instance().will_work(work)
