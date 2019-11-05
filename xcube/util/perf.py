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

import functools
import logging
import time
from contextlib import AbstractContextManager


def measure_time_cm(logger=None, disabled=False):
    """
    Get a context manager for measuring execution time of code blocks and logging the result.

    Measure duration and log output:::

        measure_time = measure_time_cm()
        with measure_time("heavy computation"):
            do_heavy_computation()

    or just measure duration:::

        with measure_time() as cm:
            do_heavy_computation()
        print("heavy computation took %2.f seconds" % cm.duration)

    :param logger: The logger to be used. May be a string or logger object. Defaults to "xcube".
    :param disabled: If True, efficiently disables timing and logging.
    :return: a context manager callable
    """
    if disabled:
        return _do_not_measure_time_cm
    else:
        return functools.partial(measure_time, logger=logger)


class measure_time(AbstractContextManager):
    def __init__(self, tag: str = None, logger=None):
        self._tag = tag
        if isinstance(logger, str):
            self._logger = logging.getLogger(logger)
        elif logger is None:
            self._logger = logging.getLogger("xcube")
        else:
            self._logger = logger
        self._start_time = None
        self.duration = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.duration = time.perf_counter() - self._start_time
        if self._tag:
            self._logger.info(self._tag + ": took " + "%.2fms" % (self.duration * 1000))


class _do_not_measure_time_cm(AbstractContextManager):

    # noinspection PyUnusedLocal
    def __init__(self, tag: str = None, logger=None):
        self.duration = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
