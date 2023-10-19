# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

from ...helpers import RoutesTestCase

from xcube.webapi.ows.coverages.routes import PATH_PREFIX


class CoveragesRoutesTest(RoutesTestCase):
    def test_fetch_coverage(self):
        response = self.fetch(
            PATH_PREFIX + '/collections/demo/coverage?f=application/json'
        )
        self.assertResponseOK(response)

    def test_fetch_coverage_html(self):
        response = self.fetch(
            PATH_PREFIX + '/collections/demo/coverage',
            headers=dict(
                Accept='text/nonexistent,application/json;q=0.9,text/html;q=1.0'
            ),
        )
        self.assertResponseOK(response)
        self.assertEqual('text/html', response.headers['Content-Type'])

    def test_fetch_coverage_wrong_media_type(self):
        response = self.fetch(
            PATH_PREFIX + '/collections/demo/coverage',
            headers=dict(Accept='text/nonexistent'),
        )
        self.assertEqual(response.status, 415)

    def test_fetch_domainset(self):
        response = self.fetch(
            PATH_PREFIX + '/collections/demo/coverage/domainset'
        )
        self.assertResponseOK(response)

    def test_fetch_rangetype(self):
        response = self.fetch(
            PATH_PREFIX + '/collections/demo/coverage/rangetype'
        )
        self.assertResponseOK(response)

    def test_fetch_metadata(self):
        response = self.fetch(
            PATH_PREFIX + '/collections/demo/coverage/metadata'
        )
        self.assertResponseOK(response)

    def test_fetch_rangeset(self):
        response = self.fetch(
            PATH_PREFIX + '/collections/demo/coverage/rangeset'
        )
        self.assertEqual(response.status, 405)
