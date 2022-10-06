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

import json
import os
import types
import unittest
from typing import Mapping, Any

import requests

from test.webapi.helpers import get_api_ctx
from xcube.server.api import ApiError
# noinspection PyProtectedMember
# noinspection PyProtectedMember
from xcube.webapi.auth.config import AuthConfig
from xcube.webapi.auth.context import AuthContext

XCUBE_TEST_CLIENT_ID = os.environ.get('XCUBE_TEST_CLIENT_ID')
XCUBE_TEST_CLIENT_SECRET = os.environ.get('XCUBE_TEST_CLIENT_SECRET')


def get_auth_ctx(config: Mapping[str, Any]) -> AuthContext:
    return get_api_ctx("auth", AuthContext, config)


class AuthContextPropsTest2(unittest.TestCase):
    def test_auth_config(self):
        auth_ctx = get_auth_ctx({
            "Authentication": {
                "Authority": "https://auth.com",
                "Audience": "myapi"
            }
        })
        auth_config = auth_ctx.auth_config
        self.assertIsInstance(auth_config, AuthConfig)
        # Assert that it is a cached property
        self.assertIs(auth_config, auth_ctx.auth_config)

    def test_granted_scopes(self):
        # Use "permissions" claim
        self.assert_granted_scopes(
            {
                "preferred_username": "bibbi",
                "permissions": [
                    "read:dataset:*~$preferred_username/*",
                    "read:variable:*"
                ]
            },
            {
                "read:dataset:*~bibbi/*",
                "read:variable:*"
            }
        )
        # Use "scope" claim
        self.assert_granted_scopes(
            {
                "preferred_username": "bibbi",
                "scope":
                    "read:dataset:*~$preferred_username/* read:variable:*"
            },
            {
                "read:dataset:*~bibbi/*",
                "read:variable:*"
            }
        )
        # Special case where "preferred_username" claim
        # is named just "username"
        self.assert_granted_scopes(
            {
                "preferred_username": "bibbi",
                "permissions": [
                    "read:dataset:*~${username}/*",
                    "read:variable:*"
                ]
            },
            {
                "read:dataset:*~bibbi/*",
                "read:variable:*"
            }
        )

    def assert_granted_scopes(self, id_token, expected_permissions):
        auth_ctx = get_auth_ctx({
            "Authentication": {
                "Authority": "https://auth.com",
                "Audience": "myapi"
            }
        })

        # noinspection PyUnusedLocal,PyShadowingNames
        def _get_id_token(self, headers, require_auth=False):
            return id_token

        auth_ctx.get_id_token = types.MethodType(_get_id_token, auth_ctx)

        self.assertEqual(expected_permissions,
                         auth_ctx.get_granted_scopes({}))


class AuthContextIdTokenTest(unittest.TestCase):
    # TODO (forman): setup Keycloak xcube test account
    @unittest.skipUnless(
        XCUBE_TEST_CLIENT_ID and XCUBE_TEST_CLIENT_SECRET,
        'XCUBE_TEST_CLIENT_ID and XCUBE_TEST_CLIENT_SECRET must be set'
    )
    def test_ok(self):
        auth_ctx = get_auth_ctx({
            "Authentication": {
                "Domain": "xcube-dev.eu.auth0.com",
                "Audience": "https://xcube-dev/api/",
            }
        })

        access_token = self._fetch_access_token()

        id_token = auth_ctx.get_id_token(
            {'Authorization': f'Bearer {access_token}'}
        )
        self.assertEqual('https://xcube-dev.eu.auth0.com/',
                         id_token.get('iss'))
        self.assertEqual('https://xcube-dev/api/',
                         id_token.get('aud'))
        self.assertEqual('w2NoDEryIpNRtAQVH1ToU6XTurE549FW',
                         id_token.get('azp'))
        self.assertEqual('client-credentials',
                         id_token.get('gty'))
        self.assertEqual('w2NoDEryIpNRtAQVH1ToU6XTurE549FW@clients',
                         id_token.get('sub'))
        self.assertEqual(['read:datasets'],
                         id_token.get('permissions'))
        exp = id_token.get('exp')
        iat = id_token.get('iat')
        self.assertIsInstance(exp, int)
        self.assertIsInstance(iat, int)
        self.assertEqual(86400, exp - iat)

    def _fetch_access_token(self):
        response = requests.post(
            'https://xcube-dev.eu.auth0.com/oauth/token',
            headers={'Content-Type': 'application/json'},
            json={
                "client_id": os.environ.get('XCUBE_TEST_CLIENT_ID'),
                "client_secret": os.environ.get('XCUBE_TEST_CLIENT_SECRET'),
                "audience": "https://xcube-dev/api/",
                "grant_type": "client_credentials"
            }
        )
        token_data = json.loads(response.content)
        if 'access_token' not in token_data:
            self.fail('failed to fetch access token for testing')
        access_token = token_data['access_token']
        return access_token

    def test_missing_auth_config(self):
        auth_ctx = get_auth_ctx({})
        self.assertIsNone(
            auth_ctx.get_id_token(
                {'Authorization': f'Bearer my_t0k3n'},
                require_auth=True
            )
        )

    def test_missing_access_token(self):
        auth_ctx = get_auth_ctx({
            "Authentication": {
                "Domain": "xcube-dev.eu.auth0.com",
                "Audience": "https://xcube-dev/api/",
            }
        })
        self.assertEqual(None, auth_ctx.get_id_token({}))

    # TODO (forman): setup Keycloak xcube test account
    @unittest.skipUnless(
        XCUBE_TEST_CLIENT_ID and XCUBE_TEST_CLIENT_SECRET,
        'XCUBE_TEST_CLIENT_ID and XCUBE_TEST_CLIENT_SECRET must be set'
    )
    def test_expired_access_token(self):
        auth_ctx = get_auth_ctx({
            "Authentication": {
                "Domain": "xcube-dev.eu.auth0.com",
                "Audience": "https://xcube-dev/api/",
            }
        })

        expired_token = \
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6" \
            "Ik5VSkdSVUV5UWpNeE16UTVRVGMzTnpRM05URkNRa1ky" \
            "TURGRE5qQkJSak5ETlRBeVFrWXdOUSJ9.eyJpc3MiOiJ" \
            "odHRwczovL3hjdWJlLWRldi5ldS5hdXRoMC5jb20vIiw" \
            "ic3ViIjoidzJOb0RFcnlJcE5SdEFRVkgxVG9VNlhUdXJ" \
            "FNTQ5RldAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8veGN" \
            "1YmUtZGV2L2FwaS8iLCJpYXQiOjE1Nzk0NTUxMDMsImV" \
            "4cCI6MTU3OTU0MTUwMywiYXpwIjoidzJOb0RFcnlJcE5" \
            "SdEFRVkgxVG9VNlhUdXJFNTQ5RlciLCJndHkiOiJjbGl" \
            "lbnQtY3JlZGVudGlhbHMiLCJwZXJtaXNzaW9ucyI6W11" \
            "9.NtGuVp0yY8foR_eFiEmH0eXvCx85cvb5b1cPWCKs6L" \
            "CNniUJJ3VnX5Fek08puW7Jidy-tj1UTzyG569dSHGDu3" \
            "10Mf7xpQ9gyfZCWcaohERxsv9MrxHziqfGVnxv051rOB" \
            "_c-fYyymrFnlsIqWeahcS7znvPoMovPO7E8MxVaIqxd_" \
            "S4zKVlH025F-bDvytuwXD-rFmYVElCg7u2uOZKqjpF3l" \
            "ZCWc50_F1jSGcEQZv4daQJY-3lfU6TnEQAuGWlOVRrCN" \
            "u05nlUBFPz6G82tB_nsP1uTa8uElOzoalVttXufLIeU0" \
            "FL8Sv-lC6wUJTZAFpykLNmpA-vhkSeTqMv4g"
        with self.assertRaises(ApiError.Unauthorized) as cm:
            auth_ctx.get_id_token(
                {'Authorization': f'Bearer {expired_token}'}
            )
        self.assertEqual('HTTP status 401: Token is expired.',
                         f'{cm.exception}')


class AuthContextGetAccessTokenTest(unittest.TestCase):

    def test_ok(self):
        headers = {'Authorization': 'Bearer my_t0k3n'}
        self.assertEqual('my_t0k3n', AuthContext.get_access_token(headers))

    def test_ok_no_auth(self):
        self.assertEqual(None, AuthContext.get_access_token({}))

    def test_not_ok(self):
        with self.assertRaises(ApiError.Unauthorized) as cm:
            AuthContext.get_access_token({}, require_auth=True)
        self.assertEqual('HTTP status 401: Authorization header is expected.',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.BadRequest) as cm:
            AuthContext.get_access_token({'Authorization': 'Beerer my_t0k3n'})
        self.assertEqual('HTTP status 400: Invalid header.'
                         ' Authorization header must start with "Bearer".',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.BadRequest) as cm:
            AuthContext.get_access_token({'Authorization': 'Bearer'})
        self.assertEqual('HTTP status 400: Invalid header.'
                         ' Bearer token not found.',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.BadRequest) as cm:
            AuthContext.get_access_token({'Authorization': 'Bearer my t0k3n'})
        self.assertEqual('HTTP status 400: Invalid header.'
                         ' Authorization header must be Bearer token.',
                         f'{cm.exception}')
