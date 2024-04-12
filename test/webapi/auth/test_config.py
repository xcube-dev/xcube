# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import pytest

from xcube.server.api import ApiError

# noinspection PyProtectedMember
from xcube.webapi.auth.config import AuthConfig


class AuthContextTest(unittest.TestCase):
    def test_from_config_empty(self):
        config = AuthConfig.from_config({})
        self.assertEqual(None, config)

    def test_from_config_ok(self):
        config = AuthConfig.from_config(
            {
                "Authentication": {
                    "Authority": "https://auth.com",
                    "Audience": "myapi",
                    "IsRequired": True,
                }
            }
        )
        self.assertEqual("https://auth.com", config.authority)
        self.assertEqual("myapi", config.audience)
        self.assertEqual(["RS256"], config.algorithms)
        self.assertEqual(True, config.is_required)
        self.assertEqual(
            "https://auth.com/.well-known/jwks.json", config.well_known_jwks
        )
        self.assertEqual(
            "https://auth.com/.well-known/openid-configuration",
            config.well_known_oid_config,
        )

    def test_from_config_with_domain_ok(self):
        config = AuthConfig.from_config(
            {
                "Authentication": {
                    "Domain": "auth.com",
                    "Audience": "myapi",
                }
            }
        )
        self.assertEqual("https://auth.com", config.authority)
        self.assertEqual("myapi", config.audience)
        self.assertEqual(["RS256"], config.algorithms)
        self.assertEqual(False, config.is_required)

    def test_from_config_fails(self):
        with pytest.raises(
            ApiError.InvalidServerConfig,
            match="HTTP status 580:"
            ' Missing key "Authority" in section'
            ' "Authentication"',
        ):
            AuthConfig.from_config({"Authentication": {}})

        with pytest.raises(
            ApiError.InvalidServerConfig,
            match="HTTP status 580:"
            ' Missing key "Audience" in section'
            ' "Authentication"',
        ):
            AuthConfig.from_config(
                {"Authentication": {"Authority": "https://auth.com"}}
            )

        with pytest.raises(
            ApiError.InvalidServerConfig,
            match="HTTP status 580:"
            ' Value for key "Algorithms" in section'
            ' "Authentication" must not be empty',
        ):
            AuthConfig.from_config(
                {
                    "Authentication": {
                        "Authority": "https://auth.com",
                        "Audience": "myapi",
                        "Algorithms": [],
                    }
                }
            )
