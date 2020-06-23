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

from typing import Sequence

import requests
from tqdm import tqdm

from xcube.cli._gen2.request import CallbackConfig
from xcube.util.assertions import assert_condition, assert_given
from xcube.util.progress import ProgressObserver
from xcube.util.progress import ProgressState


# Helge, please have a look at impl. dask.diagnostics.progress.ProgressBar
# It uses a separate Thread() to only update a progress bar within fixed time deltas.
# from dask.diagnostics.progress import ProgressBar


class CallbackApiProgressObserver(ProgressObserver):
    def __init__(self, callback_config: CallbackConfig):
        assert_given(callback_config)
        assert_condition(callback_config.api_uri and callback_config.access_token,
                         "Both, api_uri and access_token must be given.")
        self.callback_api_cfg = callback_config

    def _send_request(self, event: str, state_stack: Sequence[ProgressState]):
        assert_given(state_stack, "ProgressStates")
        state_stack = state_stack[0].to_dict()
        callback = {"event": event, "state": state_stack}

        callback_api_uri = self.callback_api_cfg.api_uri
        callback_api_access_token = self.callback_api_cfg.access_token

        header = {"Authorization": f"Bearer {callback_api_access_token}"}
        requests.put(callback_api_uri, json=callback, headers=header)

    def on_begin(self, state_stack: Sequence[ProgressState]):
        self._send_request("on_begin", state_stack)

    def on_update(self, state_stack: Sequence[ProgressState]):
        self._send_request("on_update", state_stack)

    def on_end(self, state_stack: Sequence[ProgressState]):
        self._send_request("on_end", state_stack)


class CallbackTerminalProgressObserver(ProgressObserver):
    def __init__(self):
        self._progress_bar = None

    def _print_progress(self, state_stack: [ProgressState]):
        progress_state = state_stack[0]
        if self._progress_bar:
            self._progress_bar.update(progress_state.completed_work)

    def on_begin(self, state_stack: Sequence[ProgressState]):
        self._progress_bar = tqdm(total=state_stack[0].total_work)
        self._print_progress(state_stack)

    def on_update(self, state_stack: Sequence[ProgressState]):
        self._print_progress(state_stack)

    def on_end(self, state_stack: Sequence[ProgressState]):
        self._print_progress(state_stack)
        self._progress_bar.close()
