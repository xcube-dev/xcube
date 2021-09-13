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

from xcube.util.assertions import assert_instance


class CubeGeneratorError(ValueError):
    """
    Represent a client or server error that
    may occur in the data cube generators.

    :param args: arguments passed to base exceptions.
    :param status_code: Optional status code of the error (an integer).
        If given, HTTP error codes should be used.
        The range 400-499 indicates client errors.
        The range 500-599 indicates server errors.
    :param remote_traceback: Traceback of an error
        occurred in a remote process.
    :param remote_output:  Terminal output of a remote process.
    """

    def __init__(self,
                 *args,
                 status_code: Optional[int] = None,
                 remote_traceback: Optional[List[str]] = None,
                 remote_output: Optional[List[str]] = None,
                 **kwargs):
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)
        if status_code is not None:
            assert_instance(status_code, int, 'status_code')
        self._status_code = status_code
        self._remote_traceback = remote_traceback
        self._remote_output = remote_output

    @property
    def status_code(self) -> Optional[int]:
        """
        Status code of the error.
        May be None.
        """
        return self._status_code

    @property
    def remote_traceback(self) -> Optional[List[str]]:
        """
        Traceback of an error occurred in a remote process.
        May be None.
        """
        return self._remote_traceback

    @property
    def remote_output(self) -> Optional[List[str]]:
        """
        Terminal output of a remote process.
        May be None.
        """
        return self._remote_output
