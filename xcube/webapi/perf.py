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
