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

from ..helpers import RoutesTestCase


class MetaRoutesTest(RoutesTestCase):

    def test_fetch_server_info(self):
        response = self.fetch('/')
        self.assertResponseOK(response)

    def test_fetch_openapi_json(self):
        response = self.fetch('/openapi.json')
        self.assertResponseOK(response)

    def test_fetch_openapi_html(self):
        response = self.fetch('/openapi.html')
        self.assertResponseOK(response)

    def test_fetch_maintenance_fail(self):
        response = self.fetch('/maintenance/fail')
        self.assertResponse(
            response,
            expected_status=500,
            expected_message='Error! No worries, this is just a test.'
        )

        response = self.fetch('/maintenance/fail?message=HELP')
        self.assertResponse(
            response,
            expected_status=500,
            expected_message='HELP'
        )

        response = self.fetch('/maintenance/fail?code=488')
        self.assertResponse(
            response,
            expected_status=488,
            expected_message='Error! No worries, this is just a test.'
        )

        response = self.fetch('/maintenance/fail?code=508&message=HELP')
        self.assertResponse(
            response,
            expected_status=508,
            expected_message='HELP'
        )

        response = self.fetch('/maintenance/fail?code=x')
        self.assertResponse(
            response,
            expected_status=400,
            expected_message="HTTP 400: Bad Request"
                             " (Query parameter 'code' must have type 'int'.)"
        )

    def test_fetch_maintenance_update(self):
        response = self.fetch('/maintenance/update')
        self.assertResponseOK(response)
