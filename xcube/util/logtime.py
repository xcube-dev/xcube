# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import logging
import time
from typing import Optional


class log_time:
    """Context manager that allows for logging the time
    spend to execute its context block.
    """
    def __init__(self,
                 logger: Optional[logging.Logger],
                 message: str,
                 *args, **kwargs):
        self.enabled = logger is not None \
                       and logger.isEnabledFor(logging.DEBUG)
        if self.enabled:
            self.logger = logger
            self.message = message.format(*args, **kwargs)
            self.t0 = None
            self.dt = None

    def __enter__(self):
        if self.enabled:
            self.t0 = time.time()

    def __exit__(self, *exc):
        if self.enabled:
            self.dt = time.time() - self.t0
            self.logger.debug(f'{self.message} took {int(1000 * self.dt)} ms')
