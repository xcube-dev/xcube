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

import pytest

from xcube.server.api import ApiError
# noinspection PyProtectedMember
from xcube.webapi.auth.config import AuthConfig


class AuthContextTest(unittest.TestCase):
    def test_from_config_empty(self):
        config = AuthConfig.from_config({})
        self.assertEqual(None, config)

    def test_from_config_ok(self):
        config = AuthConfig.from_config({
            "Authentication": {
                "Authority": "https://auth.com",
                "Audience": "myapi",
                "IsRequired": True,
            }
        })
        self.assertEqual("https://auth.com", config.authority)
        self.assertEqual("myapi", config.audience)
        self.assertEqual(["RS256"], config.algorithms)
        self.assertEqual(True, config.is_required)
        self.assertEqual('https://auth.com/.well-known/jwks.json',
                         config.well_known_jwks)
        self.assertEqual('https://auth.com/.well-known/openid-configuration',
                         config.well_known_oid_config)

    def test_from_config_with_domain_ok(self):
        config = AuthConfig.from_config({
            "Authentication": {
                "Domain": "auth.com",
                "Audience": "myapi",
            }
        })
        self.assertEqual("https://auth.com", config.authority)
        self.assertEqual("myapi", config.audience)
        self.assertEqual(["RS256"], config.algorithms)
        self.assertEqual(False, config.is_required)

    def test_from_config_fails(self):
        with pytest.raises(
                ApiError.InvalidServerConfig,
                match='HTTP status 580:'
                      ' Missing key "Authority" in section'
                      ' "Authentication"'
        ):
            AuthConfig.from_config({
                "Authentication": {
                }
            })

        with pytest.raises(
                ApiError.InvalidServerConfig,
                match='HTTP status 580:'
                      ' Missing key "Audience" in section'
                      ' "Authentication"'
        ):
            AuthConfig.from_config({
                "Authentication": {
                    "Authority": "https://auth.com"
                }
            })

        with pytest.raises(
                ApiError.InvalidServerConfig,
                match='HTTP status 580:'
                      ' Value for key "Algorithms" in section'
                      ' "Authentication" must not be empty'
        ):
            AuthConfig.from_config({
                "Authentication": {
                    "Authority": "https://auth.com",
                    "Audience": "myapi",
                    "Algorithms": [],
                }
            })
