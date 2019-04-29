# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

from tornado.web import HTTPError


class ServiceError(HTTPError):
    """
    Exception raised by tile service request handlers.
    """

    def __init__(self, reason: str, status_code: int = 500, log_message: str = None):
        super().__init__(status_code=status_code, log_message=log_message, reason=reason)


class ServiceConfigError(ServiceError):
    """
    Exception raised by tile service request handlers.
    """

    def __init__(self, reason: str, log_message: str = None):
        super().__init__(reason, log_message=log_message)


class ServiceBadRequestError(ServiceError):
    """
    Exception raised by tile service request handlers.
    """

    def __init__(self, reason: str, log_message: str = None):
        super().__init__(reason, status_code=400, log_message=log_message)


class ServiceResourceNotFoundError(ServiceError):
    """
    Exception raised by tile service request handlers.
    """

    def __init__(self, reason: str, log_message: str = None):
        super().__init__(reason, status_code=404, log_message=log_message)
