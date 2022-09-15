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

import unittest

from xcube.server.api import ApiError
from xcube.webapi.datasets.authutil import assert_scopes
from xcube.webapi.datasets.authutil import check_scopes


class DatasetScopesTest(unittest.TestCase):

    def test_check_scopes_ok(self):
        self.assertEqual(
            True,
            check_scopes({'read:dataset:test1.zarr'},
                         None,
                         is_substitute=True)
        )
        self.assertEqual(
            True,
            check_scopes({'read:dataset:test1.zarr'},
                         {'read:dataset:test1.zarr'})
        )
        self.assertEqual(
            True,
            check_scopes({'read:dataset:test1.zarr'},
                         {'read:dataset:test?.zarr'})
        )
        self.assertEqual(
            True,
            check_scopes({'read:dataset:test1.zarr'},
                         {'read:dataset:test1.*'})
        )

    def test_check_scopes_fails(self):
        self.assertEqual(
            False,
            check_scopes({'read:dataset:test1.zarr'},
                         None,
                         is_substitute=False)
        )
        self.assertEqual(
            False,
            check_scopes({'read:dataset:test1.zarr'},
                         set(),
                         is_substitute=True)
        )
        self.assertEqual(
            False,
            check_scopes({'read:dataset:test1.zarr'},
                         {'read:dataset:test1.zarr'},
                         is_substitute=True)
        )
        self.assertEqual(
            False,
            check_scopes({'read:dataset:test2.zarr'},
                         {'read:dataset:test1.zarr'})
        )
        self.assertEqual(
            False,
            check_scopes({'read:dataset:test2.zarr'},
                         {'read:dataset:test1.zarr'},
                         is_substitute=True)
        )

    def test_assert_scopes_ok(self):
        assert_scopes({'read:dataset:test1.zarr'},
                      {'read:dataset:test1.zarr'})
        assert_scopes({'read:dataset:test1.zarr'},
                      {'read:dataset:*'})
        assert_scopes({'read:dataset:test1.zarr'},
                      {'read:dataset:test?.zarr'})
        assert_scopes({'read:dataset:test1.zarr'},
                      {'read:dataset:test1.*'})
        assert_scopes({'read:dataset:test1.zarr'},
                      {'read:dataset:test2.zarr',
                       'read:dataset:test3.zarr',
                       'read:dataset:test1.zarr'})

    def test_assert_scopes_fails(self):
        with self.assertRaises(ApiError.Unauthorized) as cm:
            assert_scopes({'read:dataset:test1.zarr'},
                          None)
        self.assertEquals(
            'HTTP status 401: Missing permission "read:dataset:test1.zarr"',
            f'{cm.exception}'
        )

        with self.assertRaises(ApiError.Unauthorized) as cm:
            assert_scopes({'read:dataset:test1.zarr'},
                          set())
        self.assertEquals(
            'HTTP status 401: Missing permission "read:dataset:test1.zarr"',
            f'{cm.exception}'
        )

        with self.assertRaises(ApiError.Unauthorized) as cm:
            assert_scopes({'read:dataset:test1.zarr'},
                          {'read:dataset:test2.zarr'})
        self.assertEquals(
            'HTTP status 401: Missing permission "read:dataset:test1.zarr"',
            f'{cm.exception}'
        )
