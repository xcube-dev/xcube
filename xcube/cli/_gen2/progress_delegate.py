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

from xcube.cli._gen2.genconfig import CallbackConfig
from xcube.util.assertions import assert_condition, assert_given
from xcube.util.progress import ProgressState


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


def get_callback_api_progress_delegate(callback_config: CallbackConfig):
    """

    :type callback_config: CallbackConfig
    """
    assert_given(callback_config)
    assert_condition(callback_config.api_uri and callback_config.access_token,
                     "Both, api_uri and access_token must be given.")

    def _callback(sender: str, elapsed: float, state_stack: Sequence[ProgressState]):
        assert_given(state_stack, "ProgressStates")
        state = state_stack[0]
        callback = {
            "sender": sender,
            "state": {
                "label": state.label,
                "total_work": state.total_work,
                "super_work": state.super_work,
                "super_work_ahead": state.super_work_ahead,
                "exc_info": state.exc_info_text,
                "progress": state.progress,
                "elapsed": elapsed,
                "errored": state.exc_info is not None
            }
        }
        callback_api_uri = callback_config.api_uri
        callback_api_access_token = callback_config.access_token
        header = {"Authorization": f"Bearer {callback_api_access_token}"}

        return requests.put(callback_api_uri, json=callback, headers=header)

    return _callback


def get_callback_terminal_progress_delegate():
    def _callback(sender: str, elapsed: float, state_stack: [ProgressState]):
        """

        :param state_stack:
        :param sender:
        :param elapsed:
        """
        state = state_stack[0]

        bar = "#" * int(state.total_work * state.progress)
        percent = int(100 * state.progress)
        elapsed = _format_time(elapsed)
        msg = "\r{0}: [{1:<{2}}] | {3}% Completed | {4}".format(
            sender, bar, state.total_work, percent, elapsed
        )
        print(msg)

        return msg

    return _callback
