# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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
from abc import abstractmethod
from time import sleep
from timeit import default_timer
from typing import Sequence
import requests

from xcube.cli._gen2.request import CallbackConfig
from xcube.util.assertions import assert_condition, assert_given
from xcube.util.progress import ProgressObserver
from xcube.util.progress import ProgressState


# Helge, please have a look at impl. dask.diagnostics.progress.ProgressBar
# It uses a separate Thread() to only update a progress bar within fixed time deltas.


def _format_time(t):
    """Format seconds into a human readable form.
    Taken form Dask

    >>> _format_time(10.4)
    '10.4s'
    >>> _format_time(1000.4)
    '16min 40.4s'
    """
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    if h:
        return "{0:2.0f}hr {1:2.0f}min {2:4.1f}s".format(h, m, s)
    elif m:
        return "{0:2.0f}min {1:4.1f}s".format(m, s)
    else:
        return "{0:4.1f}s".format(s)


class ThreadedProgressObserver(ProgressObserver):
    """
    A threaded Progress observer adapted from Dask's ProgressBar class.
    """
    def __init__(self,
                 minimum: float = 0,
                 dt: float = 1):
        """

        :type dt: float
        :type minimum: float
        """
        super().__init__()
        assert_condition(dt >= 0, "The timer's time step must be >=0")
        assert_condition(minimum >= 0, "The timer's minimum must be >=0")
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
            self.callback(self._current_sender, elapsed, self._state_stack)
        else:
            self._update_state(elapsed)

    def _update_state(self, elapsed):
        if not self._state_stack:
            self.callback(self._current_sender, 0, elapsed)
            return
        state = self._state_stack[0]

        if state.completed_work < state.total_work:
            self.callback(self._current_sender, elapsed, self._state_stack)

    def on_begin(self, state_stack: Sequence[ProgressState]):
        assert_given(state_stack, name='state_stack')
        self._state_stack = state_stack
        self._current_sender = "on_begin"
        self._start_timer()

    def on_update(self, state_stack: Sequence[ProgressState]):
        assert_given(state_stack, name="ProgressStates")
        self._state_stack = state_stack
        self._current_sender = "on_update"

    def on_end(self, state_stack: Sequence[ProgressState]):
        assert_given(state_stack, name="ProgressStates")
        self._state_stack = state_stack
        self._current_sender = "on_end"
        self._stop_timer(False)

    @abstractmethod
    def callback(self, sender: str, elapsed: float, state_stack: Sequence[ProgressState]) -> None:
        """

        :param state_stack: Current ProgressState
        :param elapsed: elapsed time in seconds
        :param sender: sender event

        :return: None
        """


class CallbackApiProgressObserver(ThreadedProgressObserver):
    def __init__(self, callback_config: CallbackConfig):
        """

        :type callback_config: CallbackConfig
        """
        super().__init__()
        assert_given(callback_config)
        assert_condition(callback_config.api_uri and callback_config.access_token,
                         "Both, api_uri and access_token must be given.")
        self.callback_cfg = callback_config

    def callback(self, sender: str, elapsed: float, state_stack: Sequence[ProgressState]) -> None:
        assert_given(state_stack, "ProgressStates")
        callback = {"sender": sender, "state": state_stack}
        callback_api_uri = self.callback_cfg.api_uri
        callback_api_access_token = self.callback_cfg.access_token
        header = {"Authorization": f"Bearer {callback_api_access_token}"}
        requests.put(callback_api_uri, json=callback, headers=header)


class CallbackTerminalProgressObserver(ThreadedProgressObserver):
    def __init__(self,
                 minimum: float = 0,
                 dt: float = 1):
        """

        :type dt: float
        :type minimum: float
        
        """
        super().__init__(minimum, dt)

    def callback(self, sender: str, elapsed: float, state_stack: [ProgressState]):
        """

        :param state_stack:
        :param sender:
        :param elapsed:
        """
        state = state_stack[0]

        bar = "#" * int(self._width * state.progress)
        percent = int(100 * state.progress)
        elapsed = _format_time(elapsed)
        msg = "\r{0}: [{1:<{2}}] | {3}% Completed | {4}".format(
            sender, bar, self._width, percent, elapsed
        )
        print(msg)
