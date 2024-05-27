# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import functools
import logging
import time
from contextlib import AbstractContextManager
from typing import Optional

from xcube.constants import LOG


def measure_time_cm(logger=None, disabled=False):
    """Get a context manager for measuring execution time of code blocks
    and logging the result.

    Measure duration and log output:::

        measure_time = measure_time_cm()
        with measure_time("heavy computation"):
            do_heavy_computation()

    or just measure duration:::

        with measure_time() as cm:
            do_heavy_computation()
        print("heavy computation took %2.f seconds" % cm.duration)

    Args:
        logger: The logger to be used. May be a string or logger object.
            Defaults to "xcube".
        disabled: If True, efficiently disables timing and logging.

    Returns:
        a context manager callable
    """
    if disabled:
        return _do_not_measure_time_cm
    else:
        return functools.partial(measure_time, logger=logger)


class measure_time(AbstractContextManager):
    def __init__(self, *args, logger: Optional[logging.Logger] = None, **kwargs):
        self.message = args[0] if args else None
        self.args = args[1:] if args else ()
        self.kwargs = kwargs
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        elif logger is None:
            self.logger = LOG
        else:
            self.logger = logger
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.duration = time.perf_counter() - self.start_time
        if self.message:
            self.logger.info(
                self.message + ": took %.2fms",
                *self.args,
                self.duration * 1000,
                **self.kwargs
            )


class _do_not_measure_time_cm(AbstractContextManager):
    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwargs):
        self.duration = None
        self.logger = None
        self.message = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
