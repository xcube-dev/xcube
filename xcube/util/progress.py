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
from typing import Callable, Tuple, Optional, AbstractSet, Sequence

from xcube.util.assertions import assert_condition, assert_given

BeginListener = Callable[[str, float], None]
UpdateListener = Callable[[float], None]
EndListener = Callable[[], None]

ProgressMonitorTuple = Tuple[Optional[BeginListener], Optional[UpdateListener], Optional[EndListener]]


class ProgressObserver(ABC):
    """
    A progress observer is notified about nested state changes when using the
    :class:observe_progress context manager.
    """

    _active_observers = set()

    def on_begin(self, state_stack: Sequence['ProgressState']):
        """
        Called, if an observed code block begins execution.
        """
        pass

    def on_update(self, state_stack: Sequence['ProgressState']):
        """
        Called, if the progress state has changed within an observed code block.
        """
        pass

    def on_end(self, state_stack: Sequence['ProgressState']):
        """
        Called, if an observed block of code ends execution.
        """
        pass

    def activate(self):
        ProgressObserver._active_observers.add(self)

    def deactivate(self):
        ProgressObserver._active_observers.discard(self)

    @classmethod
    def active_observers(cls) -> AbstractSet['ProgressObserver']:
        return ProgressObserver._active_observers


class ProgressState:
    def __init__(self, label: str, total_work: float, super_work: float):
        self._label = label
        self._total_work = total_work
        self._super_work = super_work
        self._super_work_ahead = 1
        self._completed_work = 0
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


class new_progress_observers:
    """
    Takes zero or more progress observers and activates them in the enclosed context.
    Progress observers from an outer context will no longer be active.
    """

    def __init__(self, *observers: ProgressObserver):
        self._observers = observers

    def __enter__(self):
        ProgressObserver._active_observers, self._old_active_observers = \
            set(self._observers), ProgressObserver._active_observers

    def __exit__(self, type, value, traceback):
        ProgressObserver._active_observers = self._old_active_observers


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
    Context manager for observing progress in a block of code.
    """

    _state_stack = list()

    def __init__(self, label: str, total_work: float):
        assert_given(label, 'label')
        assert_condition(total_work > 0, 'total_work must be greater than zero')
        self.label = label
        self.total_work = total_work

    def __enter__(self) -> 'observe_progress':
        super_work = self._state_stack[-1].super_work_ahead if self._state_stack else 1
        self._state_stack.append(ProgressState(self.label, self.total_work, super_work))
        self._emit_begin()
        return self

    def __exit__(self, type, value, traceback):
        # store error info
        self._state_stack[-1].finish()
        self._emit_end()
        self._state_stack.pop()
        if self._state_stack:
            self._state_stack[-1].super_work_ahead = 1

    def worked(self, work: float):
        assert_condition(self._state_stack, 'worked() method call is missing a current context')
        assert_condition(work > 0, 'work must be greater than zero')
        for s in reversed(self._state_stack):
            s.inc_work(work)
            work = s.to_super_work(work)
        self._emit_update()

    def will_work(self, work: float):
        assert_condition(self._state_stack, 'will_work() method call is missing a current context')
        # noinspection PyProtectedMember
        self._state_stack[-1].super_work_ahead = work

    def _emit_begin(self):
        for observer in ProgressObserver.active_observers():
            observer.on_begin(self._state_stack)

    def _emit_update(self):
        for observer in ProgressObserver.active_observers():
            observer.on_update(self._state_stack)

    def _emit_end(self):
        for observer in ProgressObserver.active_observers():
            observer.on_end(self._state_stack)


class observe_progress_local(observe_progress):
    """
    Context manager for locally observing progress in a block of code.
    """

    def __init__(self, label: str, total_work: float):
        super().__init__(label, total_work)
        self._old_state_stack = None

    def __enter__(self) -> 'observe_progress':
        observe_progress._state_stack, self._old_state_stack = list(), observe_progress._state_stack
        super().__enter__()
        return self

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)
        observe_progress._state_stack = self._old_state_stack
        self._old_state_stack = None
