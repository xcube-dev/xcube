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

from xcube.util.progress import ProgressObserver
from xcube.util.progress import ProgressState


# Helge, please have a look at impl. dask.diagnostics.progress.ProgressBar
# It uses a separate Thread() to only update a progress bar within fixed time deltas.
# from dask.diagnostics.progress import ProgressBar


class CallbackApiProgressObserver(ProgressObserver):
    def __init__(self, callback_api_url: str):
        self.callback_api_url = callback_api_url

    def on_begin(self, state_stack: Sequence[ProgressState]):
        # TODO (dzelge): implement me
        pass

    def on_update(self, state_stack: Sequence[ProgressState]):
        # TODO (dzelge): implement me
        pass

    def on_end(self, state_stack: Sequence[ProgressState]):
        # TODO (dzelge): implement me
        pass
