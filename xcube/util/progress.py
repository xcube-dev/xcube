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
from typing import Callable, Tuple, Optional, Union

BeginCallback = Callable[[str, float], None]
UpdateCallback = Callable[[float], None]
EndCallback = Callable[[], None]

ProgressMonitorTuple = Tuple[Optional[BeginCallback], Optional[UpdateCallback], Optional[EndCallback]]

class ProgressMonitor:
    ACTIVE = set()

    def __init__(self, begin: BeginCallback = None, update: UpdateCallback = None, end: EndCallback = None):
        if begin:
            self.begin = begin
        if update:
            self.update = update
        if end:
            self.end = end

    def _to_tuple(self) -> ProgressMonitorTuple:
        return getattr(self, 'begin', None), getattr(self, 'update', None), getattr(self, 'end', None)

    @classmethod
    def normalize(cls, monitor: Union[ProgressMonitorTuple, 'ProgressMonitor']) -> ProgressMonitorTuple:
        if isinstance(monitor, ProgressMonitor):
            return monitor._to_tuple()
        if isinstance(monitor, tuple):
            return monitor
        raise TypeError('monitor must bei either a ProgressMonitor or a tuple of callables')

    @classmethod
    def _notify_begin(cls, label: str, total_work: float):
        cls._notify('begin', label, total_work)

    @classmethod
    def _notify_update(cls, worked: float):
        cls._notify('update', worked)

    @classmethod
    def _notify_end(cls):
        cls._notify('end')

    @classmethod
    def _notify(cls, callback_name, *args, **kwargs):
        for pm in cls.ACTIVE.values():
            callback = getattr(pm, callback_name, None)
            if callback is not None:
                callback(*args, **kwargs)

    def register(self):
        ProgressMonitor.ACTIVE.update(self._to_tuple())

    def deregister(self):
        ProgressMonitor.ACTIVE.discard(self._to_tuple())


class ProgressEmitter:
    def __init__(self, label: str, total_work: float):
        self._label = label
        self._total_work = total_work
        self._worked = 0.0
        self._stopped = False

    def update(self, worked: float):
        self._worked = worked

    def stop(self):
        self._stopped = True

    def __enter__(self):
        self._cm = add_monitors(self)
        self._cm.__enter__()
        return self

    def __exit__(self, *args):
        self._cm.__exit__(*args)

class add_monitors:
    """Context manager for monitors.

    Takes several callbacks and applies them only in the enclosed context.
    Monitors can either be represented as a ``ProgressMonitor`` object, or as a tuple
    of length 4.

    Examples
    --------
    >>> def update(worked):
    ...     print(f"Worked: {worked}")
    >>> monitors = (None, update, None, None)
    >>> with add_monitors(monitors):    # doctest: +SKIP
    ...     res.compute()
    """

    def __init__(self, *monitors: ProgressMonitor):
        self._monitors = [ProgressMonitor.normalize(m) for m in monitors]

    def __enter__(self):
        ProgressMonitor.ACTIVE.update(self._monitors)

    def __exit__(self, type, value, traceback):
        for c in self._monitors:
            ProgressMonitor.ACTIVE.discard(c)
