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

import os
from typing import Sequence

from xcube.server.server import Server
from xcube.util.config import load_configs


class ConfigChangeObserver:
    """
    An observer for configuration changes.
    :param server: The server
    :param config_paths: Configuration file paths.
    :param check_after: Time in seconds between two observations.
    """

    def __init__(self,
                 server: Server,
                 config_paths: Sequence[str],
                 check_after: float):
        self._server = server
        self._config_paths = config_paths
        self._check_after = check_after
        self._last_stats = None

    def check(self):
        last_stats = self._last_stats
        next_stats = [os.stat(config_path)
                      for config_path in self._config_paths]
        if self._change_detected(last_stats, next_stats):
            next_config = load_configs(*self._config_paths)
            self._server.update(next_config)
        self._last_stats = next_stats
        self._server.call_later(self._check_after, self.check)

    @staticmethod
    def _change_detected(last_stats, next_stats) -> bool:
        if last_stats:
            for s1, s2 in zip(next_stats, last_stats):
                if s1.st_mtime != s2.st_mtime:
                    return True
        return False
