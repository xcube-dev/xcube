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
import os.path
import threading
from typing import Any, Dict, List, Optional

from xcube.constants import LOG
from xcube.server.api import ApiContext
from xcube.server.api import Context
from xcube.server.api import ServerConfigObject
from xcube.util.perf import measure_time_cm
from xcube.version import version
from xcube.webapi._auth import AuthContext
from xcube.webapi.errors import ServiceError


class ResourcesContext(ApiContext):

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        # noinspection PyTypeChecker
        self._auth_ctx: AuthContext = server_ctx.get_api_ctx("_auth")
        assert isinstance(self._auth_ctx, AuthContext)
        self._base_dir = os.path.abspath(self.config.get("base_dir", "."))
        self._prefix = normalize_prefix(self.config.get("prefix", ""))
        self._trace_perf = self.config.get("trace_perf", False)
        self._rlock = threading.RLock()

    @property
    def auth_ctx(self) -> AuthContext:
        return self._auth_ctx

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def rlock(self) -> threading.RLock:
        return self._rlock

    @property
    def trace_perf(self) -> bool:
        return self._trace_perf

    @property
    def measure_time(self):
        return measure_time_cm(disabled=not self.trace_perf, logger=LOG)

    @property
    def can_authenticate(self) -> bool:
        """
        Test whether the user can authenticate.
        Even if authentication service is configured, user authentication
        may still be optional. In this case the server will publish
        the resources configured to be free for everyone.
        """
        return self._auth_ctx.can_authenticate

    @property
    def must_authenticate(self) -> bool:
        """
        Test whether the user must authenticate.
        """
        return self._auth_ctx.must_authenticate

    @property
    def access_control(self) -> Dict[str, Any]:
        return self.config.get('AccessControl', {})

    @property
    def required_scopes(self) -> List[str]:
        return self.access_control.get('RequiredScopes', [])

    def get_service_url(self, base_url: Optional[str], *path: str):
        base_url = base_url or ''
        # noinspection PyTypeChecker
        path_comp = '/'.join(path)
        if self._prefix:
            return base_url + self._prefix + '/' + path_comp
        else:
            return base_url + '/' + path_comp

    def get_config_path(self,
                        config: ServerConfigObject,
                        config_name: str,
                        path_entry_name: str = 'Path',
                        is_url: bool = False) -> str:
        path = config.get(path_entry_name)
        if not path:
            raise ServiceError(
                f"Missing entry {path_entry_name!r} in {config_name}")
        if not is_url and not os.path.isabs(path):
            path = os.path.join(self._base_dir, path)
        return path


def normalize_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ''

    prefix = prefix.replace('${name}', 'xcube')
    prefix = prefix.replace('${version}', version)
    prefix = prefix.replace('//', '/').replace('//', '/')

    if prefix == '/':
        return ''

    if not prefix.startswith('/'):
        prefix = '/' + prefix

    if prefix.endswith('/'):
        prefix = prefix[0:-1]

    return prefix
