import json
import os
import unittest

import requests

from xcube.webapi.auth import AuthMixin
from xcube.webapi.auth import assert_scopes
from xcube.webapi.auth import check_scopes
from xcube.webapi.errors import ServiceAuthError
from xcube.webapi.errors import ServiceConfigError


class ServiceContextMock:
    def __init__(self, config):
        self.config = config


class RequestMock:
    def __init__(self, headers):
        self.headers = headers


XCUBE_TEST_CLIENT_ID = os.environ.get('XCUBE_TEST_CLIENT_ID')
XCUBE_TEST_CLIENT_SECRET = os.environ.get('XCUBE_TEST_CLIENT_SECRET')


class AuthMixinIdTokenTest(unittest.TestCase):

    @unittest.skipUnless(
        XCUBE_TEST_CLIENT_ID and XCUBE_TEST_CLIENT_SECRET,
        'XCUBE_TEST_CLIENT_ID and XCUBE_TEST_CLIENT_SECRET must be set'
    )
    def test_ok(self):
        access_token = self._fetch_access_token()

        auth_mixin = AuthMixin()
        auth_mixin.service_context = ServiceContextMock(config=dict(
            Authentication=dict(
                Domain='xcube-dev.eu.auth0.com',
                Audience='https://xcube-dev/api/'
            )))
        auth_mixin.request = RequestMock(
            headers={'Authorization': f'Bearer {access_token}'}
        )

        id_token = auth_mixin.get_id_token()
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
        auth_mixin = AuthMixin()
        auth_mixin.service_context = ServiceContextMock(config={})
        auth_mixin.request = RequestMock(
            headers={'Authorization': f'Bearer my_t0k3n'}
        )

        with self.assertRaises(ServiceAuthError) as cm:
            auth_mixin.get_id_token(require_auth=True)

        self.assertEqual('HTTP 401: Invalid header (Received access token, '
                         'but this server doesn\'t support authentication.)',
                         f'{cm.exception}')

    def test_missing_access_token(self):
        auth_mixin = AuthMixin()
        auth_mixin.request = RequestMock(headers={})
        self.assertEqual(None, auth_mixin.get_id_token())

    # @unittest.skipUnless(
    #     XCUBE_TEST_CLIENT_ID and XCUBE_TEST_CLIENT_SECRET,
    #     'XCUBE_TEST_CLIENT_ID and XCUBE_TEST_CLIENT_SECRET must be set'
    # )
    def test_expired_access_token(self):
        auth_mixin = AuthMixin()
        auth_mixin.service_context = ServiceContextMock(config=dict(
            Authentication=dict(
                Domain='xcube-dev.eu.auth0.com',
                Audience='https://xcube-dev/api/'
            )))

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
        auth_mixin.request = RequestMock(
            headers={'Authorization': f'Bearer {expired_token}'})
        with self.assertRaises(ServiceAuthError) as cm:
            auth_mixin.get_id_token()
        self.assertEqual('HTTP 401: Token expired (Token is expired)',
                         f'{cm.exception}')


class AuthMixinAccessTokenTest(unittest.TestCase):

    def test_ok(self):
        auth_mixin = AuthMixin()
        auth_mixin.request = RequestMock(
            headers={'Authorization': 'Bearer my_t0k3n'}
        )
        self.assertEqual('my_t0k3n', auth_mixin.get_access_token())

    def test_ok_no_auth(self):
        auth_mixin = AuthMixin()
        auth_mixin.request = RequestMock(headers={})
        self.assertEqual(None, auth_mixin.get_access_token())

    def test_not_ok(self):
        auth_mixin = AuthMixin()

        auth_mixin.request = RequestMock(headers={})
        with self.assertRaises(ServiceAuthError) as cm:
            auth_mixin.get_access_token(require_auth=True)
        self.assertEqual('HTTP 401: Authorization header'
                         ' missing (Authorization header is expected)',
                         f'{cm.exception}')

        auth_mixin.request = RequestMock(
            headers={'Authorization': 'Beerer my_t0k3n'}
        )
        with self.assertRaises(ServiceAuthError) as cm:
            auth_mixin.get_access_token()
        self.assertEqual('HTTP 401: Invalid header'
                         ' (Authorization header must start with "Bearer")',
                         f'{cm.exception}')

        auth_mixin.request = RequestMock(
            headers={'Authorization': 'Bearer'}
        )
        with self.assertRaises(ServiceAuthError) as cm:
            auth_mixin.get_access_token()
        self.assertEqual('HTTP 401: Invalid header (Bearer token not found)',
                         f'{cm.exception}')

        auth_mixin.request = RequestMock(
            headers={'Authorization': 'Bearer my t0k3n'}
        )
        with self.assertRaises(ServiceAuthError) as cm:
            auth_mixin.get_access_token()
        self.assertEqual('HTTP 401: Invalid header'
                         ' (Authorization header must be Bearer token)',
                         f'{cm.exception}')


class AuthMixinAuthConfigTest(unittest.TestCase):

    def test_ok(self):
        auth_mixin = AuthMixin()
        auth_mixin.service_context = ServiceContextMock(config=dict(
            Authentication=dict(
                Domain='xcube-dev.eu.auth0.com',
                Audience='https://xcube-dev/api/'
            )))
        auth_config = auth_mixin.auth_config
        self.assertEqual('xcube-dev.eu.auth0.com',
                         auth_config.domain)
        self.assertEqual('https://xcube-dev/api/',
                         auth_config.audience)
        self.assertEqual('https://xcube-dev.eu.auth0.com/',
                         auth_config.issuer)
        self.assertEqual('https://xcube-dev.eu.auth0.com/'
                         '.well-known/jwks.json',
                         auth_config.well_known_jwks)
        self.assertEqual(['RS256'], auth_config.algorithms)

    def test_not_ok(self):
        # class ServiceContextMock:
        #     def __init__(self, config):
        #         self.config = config

        auth_mixin = AuthMixin()
        auth_mixin.service_context = ServiceContextMock(config=dict(
            Authentication=dict(
                Domain='xcube-dev.eu.auth0.com',
            )))

        with self.assertRaises(ServiceConfigError) as cm:
            # noinspection PyUnusedLocal
            auth_config = auth_mixin.auth_config
        self.assertEqual('HTTP 500: Missing key'
                         ' "Audience" in section "Authentication"',
                         f'{cm.exception}')

        auth_mixin.service_context = ServiceContextMock(config=dict(
            Authentication=dict(
                Audience='https://xcube-dev/api/'
            )))

        with self.assertRaises(ServiceConfigError) as cm:
            # noinspection PyUnusedLocal
            auth_config = auth_mixin.auth_config
        self.assertEqual('HTTP 500: Missing'
                         ' key "Domain" in section "Authentication"',
                         f'{cm.exception}')

        auth_mixin.service_context = ServiceContextMock(config=dict(
            Authentication=dict(
                Domain='xcube-dev.eu.auth0.com',
                Audience='https://xcube-dev/api/',
                Algorithms=[],
            )))

        with self.assertRaises(ServiceConfigError) as cm:
            # noinspection PyUnusedLocal
            auth_config = auth_mixin.auth_config
        self.assertEqual('HTTP 500: Value for key "Algorithms"'
                         ' in section "Authentication" must not be empty',
                         f'{cm.exception}')


class ScopesTest(unittest.TestCase):

    def test_check_scopes_ok(self):
        self.assertEqual(
            True,
            check_scopes({'read:dataset:test1.zarr'},
                         set(),
                         is_substitute=False)
        )
        self.assertEqual(
            True,
            check_scopes({'read:dataset:test1.zarr'},
                         set(),
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
                      set())
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
        with self.assertRaises(ServiceAuthError) as cm:
            assert_scopes({'read:dataset:test1.zarr'},
                          {'read:dataset:test2.zarr'})
        self.assertEquals(
            'HTTP 401: Missing permission'
            ' (Missing permission read:dataset:test1.zarr)',
            f'{cm.exception}'
        )
