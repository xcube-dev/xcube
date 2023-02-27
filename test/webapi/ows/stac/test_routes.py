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

from ...helpers import RoutesTestCase


class StacRoutesTest(RoutesTestCase):
    """STAC endpoints smoke tests"""

    def test_fetch_catalog(self):
        response = self.fetch('/catalog')
        self.assertResponseOK(response)

    def test_fetch_catalog_conformance(self):
        response = self.fetch('/catalog/conformance')
        self.assertResponseOK(response)

    def test_fetch_catalog_collections(self):
        response = self.fetch('/catalog/collections')
        self.assertResponseOK(response)

    def test_fetch_catalog_collection(self):
        response = self.fetch('/catalog/collections/datasets')
        self.assertResponseOK(response)
        response = self.fetch('/catalog/collections/datacubes')
        self.assertResourceNotFoundResponse(response)

    def test_fetch_catalog_collection_items(self):
        response = self.fetch('/catalog/collections/datasets/items')
        self.assertResponseOK(response)
        response = self.fetch('/catalog/collections/datacubes')
        self.assertResourceNotFoundResponse(response)

    def test_fetch_catalog_collection_item(self):
        response = self.fetch('/catalog/collections/datasets/items/demo')
        self.assertResponseOK(response)
        response = self.fetch('/catalog/collections/datasets/items/demox')
        self.assertResourceNotFoundResponse(response)
        response = self.fetch('/catalog/collections/datacubes/items/demo')
        self.assertResourceNotFoundResponse(response)

    def test_fetch_catalog_search_by_kw(self):
        response = self.fetch('/catalog/search', method='GET')
        self.assertResponseOK(response)

    def test_fetch_catalog_search_by_json(self):
        response = self.fetch('/catalog/search', method='POST')
        self.assertResponseOK(response)
