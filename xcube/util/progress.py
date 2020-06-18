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

from abc import ABC
from typing import Sequence

from xcube.util.assertions import assert_condition, assert_given


class ProgressState:
    """Represents the state of progress."""

    def __init__(self, label: str, total_work: float, super_work: float):
        self._label = label
        self._total_work = total_work
        self._super_work = super_work
        self._super_work_ahead = 1.
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

    def end(self, type, value, traceback):
        # store error info
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
