# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.server.api import ApiError
from xcube.webapi.datasets.authutil import assert_scopes
from xcube.webapi.datasets.authutil import check_scopes


class DatasetScopesTest(unittest.TestCase):
    def test_check_scopes_ok(self):
        self.assertEqual(
            True, check_scopes({"read:dataset:test1.zarr"}, None, is_substitute=True)
        )
        self.assertEqual(
            True, check_scopes({"read:dataset:test1.zarr"}, {"read:dataset:test1.zarr"})
        )
        self.assertEqual(
            True, check_scopes({"read:dataset:test1.zarr"}, {"read:dataset:test?.zarr"})
        )
        self.assertEqual(
            True, check_scopes({"read:dataset:test1.zarr"}, {"read:dataset:test1.*"})
        )

    def test_check_scopes_fails(self):
        self.assertEqual(
            False, check_scopes({"read:dataset:test1.zarr"}, None, is_substitute=False)
        )
        self.assertEqual(
            False, check_scopes({"read:dataset:test1.zarr"}, set(), is_substitute=True)
        )
        self.assertEqual(
            False,
            check_scopes(
                {"read:dataset:test1.zarr"},
                {"read:dataset:test1.zarr"},
                is_substitute=True,
            ),
        )
        self.assertEqual(
            False,
            check_scopes({"read:dataset:test2.zarr"}, {"read:dataset:test1.zarr"}),
        )
        self.assertEqual(
            False,
            check_scopes(
                {"read:dataset:test2.zarr"},
                {"read:dataset:test1.zarr"},
                is_substitute=True,
            ),
        )

    def test_assert_scopes_ok(self):
        assert_scopes({"read:dataset:test1.zarr"}, {"read:dataset:test1.zarr"})
        assert_scopes({"read:dataset:test1.zarr"}, {"read:dataset:*"})
        assert_scopes({"read:dataset:test1.zarr"}, {"read:dataset:test?.zarr"})
        assert_scopes({"read:dataset:test1.zarr"}, {"read:dataset:test1.*"})
        assert_scopes(
            {"read:dataset:test1.zarr"},
            {
                "read:dataset:test2.zarr",
                "read:dataset:test3.zarr",
                "read:dataset:test1.zarr",
            },
        )

    def test_assert_scopes_fails(self):
        with self.assertRaises(ApiError.Unauthorized) as cm:
            assert_scopes({"read:dataset:test1.zarr"}, None)
        self.assertEqual(
            'HTTP status 401: Missing permission "read:dataset:test1.zarr"',
            f"{cm.exception}",
        )

        with self.assertRaises(ApiError.Unauthorized) as cm:
            assert_scopes({"read:dataset:test1.zarr"}, set())
        self.assertEqual(
            'HTTP status 401: Missing permission "read:dataset:test1.zarr"',
            f"{cm.exception}",
        )

        with self.assertRaises(ApiError.Unauthorized) as cm:
            assert_scopes({"read:dataset:test1.zarr"}, {"read:dataset:test2.zarr"})
        self.assertEqual(
            'HTTP status 401: Missing permission "read:dataset:test1.zarr"',
            f"{cm.exception}",
        )
