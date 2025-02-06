# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os
from collections.abc import Sequence

from xcube.server.server import Server
from xcube.util.config import load_configs


class ConfigChangeObserver:
    """An observer for configuration changes.

    Args:
        server: The server
        config_paths: Configuration file paths.
        check_after: Time in seconds between two observations.
    """

    def __init__(self, server: Server, config_paths: Sequence[str], check_after: float):
        self._server = server
        self._config_paths = config_paths
        self._check_after = check_after
        self._last_stats = None

    def check(self):
        last_stats = self._last_stats
        next_stats = [os.stat(config_path) for config_path in self._config_paths]
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
