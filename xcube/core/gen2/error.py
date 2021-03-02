# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

from typing import Optional, List

import requests


class CubeGeneratorError(ValueError):
    def __init__(self, *args,
                 remote_traceback: str = None,
                 remote_output: List[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._remote_traceback = remote_traceback
        self._remote_output = remote_output

    @property
    def remote_traceback(self) -> Optional[str]:
        """Traceback of an error occurred in a remote process."""
        return self._remote_traceback

    @property
    def remote_output(self) -> Optional[List[str]]:
        """Terminal output of a remote process."""
        return self._remote_output

    @classmethod
    def maybe_raise_for_response(cls, response: requests.Response):
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            detail = None
            traceback = None
            # noinspection PyBroadException
            try:
                json = response.json()
                if isinstance(json, dict):
                    detail = json.get('detail')
                    traceback = json.get('traceback')
            except Exception:
                pass
            raise CubeGeneratorError(f'{e}: {detail}' if detail else f'{e}',
                                     remote_traceback=traceback) from e
