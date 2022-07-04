# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

import fnmatch
import json
from functools import cached_property
from itertools import filterfalse
from string import Template
from typing import Optional, Mapping, List, Dict, Any, Set, Union

import jwt
import requests
from jwt.algorithms import RSAAlgorithm

from xcube.constants import LOG
from xcube.webapi.errors import ServiceAuthError, ServiceConfigError

READ_ALL_DATASETS_SCOPE = 'read:dataset:*'
READ_ALL_VARIABLES_SCOPE = 'read:variable:*'


class AuthConfig:
    def __init__(self,
                 authority: str,
                 audience: str,
                 algorithms: List[str],
                 is_required: bool = False):
        self._authority = authority
        self._audience = audience
        self._algorithms = algorithms
        self._is_required = is_required

    @property
    def authority(self) -> str:
        return self._authority

    @property
    def _norm_authority(self) -> str:
        return self.authority[:-1] \
            if self.authority.endswith('/') \
            else self.authority

    @property
    def well_known_oid_config(self) -> str:
        return f"{self._norm_authority}/.well-known/openid-configuration"

    @property
    def well_known_jwks(self) -> str:
        return f"{self._norm_authority}/.well-known/jwks.json"

    @property
    def audience(self) -> str:
        return self._audience

    @property
    def algorithms(self) -> List[str]:
        return self._algorithms

    @property
    def is_required(self) -> bool:
        return self._is_required

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional['AuthConfig']:
        authentication = config.get('Authentication')
        if not authentication:
            return None
        authority = authentication.get('Authority')
        domain = authentication.get('Domain')
        if domain:
            LOG.warning('Configuration parameter "Domain"'
                        ' in section "Authentication"'
                        ' has been deprecated. Use "Authority" instead.')
            if not authority:
                authority = f"https:/{domain}"
        if not authority:
            raise ServiceConfigError(
                'Missing key "Authority" in section "Authentication"'
            )
        audience = authentication.get('Audience')
        if not audience:
            raise ServiceConfigError(
                'Missing key "Audience" in section "Authentication"'
            )
        algorithms = authentication.get('Algorithms', ['RS256'])
        if not algorithms:
            raise ServiceConfigError(
                'Value for key "Algorithms" in section'
                ' "Authentication" must not be empty'
            )
        is_required = authentication.get('IsRequired', False)
        return AuthConfig(authority,
                          audience,
                          algorithms,
                          is_required=is_required)


class AuthMixin:
    """
    To be mixed with a class that provides a property
    'service_context' of type ServiceContext and
    'request' of type tornado.Request.
    """

    @cached_property
    def auth_config(self) -> Optional[AuthConfig]:
        assert hasattr(self, 'service_context')
        # noinspection PyUnresolvedReferences
        return AuthConfig.from_config(self.service_context.config)

    @cached_property
    def jwks(self):
        jwks_uri = self.auth_config.well_known_jwks
        openid_config_uri = self.auth_config.well_known_oid_config
        response = requests.get(openid_config_uri)
        if response.ok:
            openid_config = json.loads(response.content)
            if openid_config and 'jwks_uri' in openid_config:
                jwks_uri = openid_config['jwks_uri']
        response = requests.get(jwks_uri)
        if response.ok:
            return json.loads(response.content)
        response.raise_for_status()

    @property
    def granted_scopes(self) -> Optional[Set[str]]:
        assert hasattr(self, 'service_context')
        # noinspection PyUnresolvedReferences
        must_authenticate = self.service_context.must_authenticate
        id_token = self.get_id_token(require_auth=must_authenticate)
        permissions = None
        if id_token:
            permissions = id_token.get('permissions')
            if not isinstance(permissions, (list, tuple)):
                scope = id_token.get('scope')
                if isinstance(scope, str):
                    permissions = scope.split(' ')
            if permissions is not None:
                permissions = self._interpolate_permissions(id_token,
                                                            permissions)
        return permissions

    def _interpolate_permissions(self,
                                 id_token: Mapping[str, Any],
                                 permissions: Union[list, tuple]):
        predicate = self._is_template_permission

        plain_permissions = set(filterfalse(predicate, permissions))
        if len(plain_permissions) == len(permissions):
            return plain_permissions

        templ_permissions = filter(predicate, permissions)
        id_mapping = self._get_template_dict(id_token)
        return plain_permissions.union(
            set(Template(permission).safe_substitute(id_mapping)
                for permission in templ_permissions)
        )

    @staticmethod
    def _is_template_permission(permission: str) -> bool:
        return '$' in permission

    @staticmethod
    def _get_template_dict(id_token: Mapping[str, Any]) -> Dict[str, str]:
        d = {k: v
             for k, v in id_token.items()
             if isinstance(v, str)}
        if 'username' not in d and 'preferred_username' in d:
            d['username'] = d['preferred_username']
        return d

    def get_id_token(self, require_auth: bool = False) \
            -> Optional[Mapping[str, Any]]:
        """
        Converts the request's access token into an id token.
        """
        auth_config = self.auth_config
        if auth_config is None:
            if require_auth:
                raise ServiceAuthError(
                    "Invalid header",
                    log_message="Received access token,"
                                " but this server doesn't support"
                                " authentication."
                )
            return None

        access_token = self.get_access_token(require_auth=require_auth)
        if access_token is None:
            return None

        # With auth_config and access_token, we expect authorization
        # to work. From here on we raise, if anything fails.

        # Get the unverified header of the access token
        try:
            unverified_header = jwt.get_unverified_header(access_token)
        except jwt.InvalidTokenError:
            unverified_header = None
        if not unverified_header \
                or not unverified_header.get("kid") \
                or not unverified_header.get("alg"):
            # "alg" should be "RS256" or "HS256" or others
            raise ServiceAuthError(
                "Invalid header",
                log_message="Invalid header."
                            " An signed JWT Access Token is expected."
            )

        # The key identifier of the access token which we must validate.
        access_token_kid = unverified_header["kid"]

        # Get JSON Web Token (JWK) Keys
        jwks = self.jwks

        # Find access_token_kid in JWKS to obtain rsa_key
        rsa_key = None
        for key in jwks["keys"]:
            if key["kid"] == access_token_kid:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break
        if rsa_key is None:
            raise ServiceAuthError(
                "Invalid header",
                log_message="Unable to find appropriate key."
            )

        # Now we are ready to decode the access token
        try:
            id_token = jwt.decode(
                access_token,
                RSAAlgorithm.from_jwk(rsa_key),
                issuer=auth_config.authority,
                audience=auth_config.audience,
                algorithms=auth_config.algorithms
            )
        except jwt.PyJWTError as e:
            msg = f"Failed to decode access token: {e}"
            raise ServiceAuthError(msg, log_message=msg) from e

        return id_token

    def get_access_token(self, require_auth: bool = False) -> Optional[str]:
        """
        Obtain the request's access token from the Authorization Header.
        """
        # noinspection PyUnresolvedReferences
        auth = self.request.headers.get("Authorization", None)
        if not auth:
            if require_auth:
                raise ServiceAuthError(
                    "Authorization header missing",
                    log_message="Authorization header is expected"
                )
            return None

        parts = auth.split()

        if parts[0].lower() != "bearer":
            raise ServiceAuthError(
                "Invalid header",
                log_message='Authorization header must start with "Bearer"'
            )
        elif len(parts) == 1:
            raise ServiceAuthError(
                "Invalid header",
                log_message="Bearer token not found"
            )
        elif len(parts) > 2:
            raise ServiceAuthError(
                "Invalid header",
                log_message="Authorization header must be Bearer token"
            )

        return parts[1]


def assert_scopes(required_scopes: Set[str],
                  granted_scopes: Optional[Set[str]],
                  is_substitute: bool = False):
    """
    Assert scopes.
    Raise ServiceAuthError if one of *required_scopes* is
    not in *granted_scopes*.

    :param required_scopes: The list of required scopes
    :param granted_scopes: The set of granted scopes.
        If user is not authenticated, its value is None.
    :param is_substitute: True, if the resource to be checked
        is a substitute.
    """
    missing_scope = _get_missing_scope(required_scopes,
                                       granted_scopes,
                                       is_substitute=is_substitute)
    if missing_scope is not None:
        raise ServiceAuthError(
            'Missing permission',
            log_message=f'Missing permission {missing_scope}'
        )


def check_scopes(required_scopes: Set[str],
                 granted_scopes: Optional[Set[str]],
                 is_substitute: bool = False) -> bool:
    """
    Check scopes.

    This function is used to filter out a resource's sub-resources for
    which a given client has no permission.

    If one of *required_scopes* is not in *granted_scopes*, fail.
    If *granted_scopes* exists and *is_substitute*, fail too.
    Else succeed.

    :param required_scopes: The list of required scopes
    :param granted_scopes: The set of granted scopes.
        If user is not authenticated, its value is None.
    :param is_substitute: True, if the resource to be checked
        is a substitute.
    :return: True, if scopes are ok.
    """
    return _get_missing_scope(required_scopes,
                              granted_scopes,
                              is_substitute=is_substitute) is None


def _get_missing_scope(required_scopes: Set[str],
                       granted_scopes: Optional[Set[str]],
                       is_substitute: bool = False) -> Optional[str]:
    """
    Return the first required scope that is
    fulfilled by any granted scope

    :param required_scopes: The list of required scopes
    :param granted_scopes: The set of granted scopes.
        If user is not authenticated, its value is None.
    :param is_substitute: True, if the resource to be checked
        is a substitute.
    :return: The missing scope.
    """
    is_authenticated = granted_scopes is not None
    if is_authenticated:
        for required_scope in required_scopes:
            required_permission_given = False
            for granted_scope in granted_scopes:
                if required_scope == granted_scope \
                        or fnmatch.fnmatch(required_scope, granted_scope):
                    # If any granted scope matches, we can stop
                    required_permission_given = True
                    break
            if not required_permission_given:
                # The required scope is not a granted scope --> fail
                return required_scope

        # If we end here, required_scopes are either empty or satisfied
        if is_substitute:
            # All required scopes are satisfied, now fail for
            # substitute resources (e.g. demo resources) as there
            # is usually a better (non-demo) resource that replaces it.
            # Return missing scope (dummy, not used) --> fail
            return READ_ALL_DATASETS_SCOPE

    elif required_scopes:
        # We require scopes but have no granted scopes
        if not is_substitute:
            # ...and resource is not a substitute --> fail
            return next(iter(required_scopes))

    # All ok.
    return None
