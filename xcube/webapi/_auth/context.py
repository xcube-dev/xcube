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
from functools import cached_property
from typing import Optional, Mapping, List, Set, Dict, Any

import jwt
import jwt.algorithms
import requests

from xcube.server.api import ApiContext
from xcube.webapi.errors import ServiceAuthError, ServiceConfigError

READ_ALL_DATASETS_SCOPE = 'read:dataset:*'
READ_ALL_VARIABLES_SCOPE = 'read:variable:*'


class AuthContext(ApiContext):

    @cached_property
    def authentication(self) -> Optional[dict]:
        return self.config.get('Authentication') or None

    @cached_property
    def domain(self) -> str:
        authentication = self.authentication
        assert isinstance(authentication, dict)
        domain = authentication.get('Domain')
        if not domain:
            raise ServiceConfigError(
                'Missing key "Domain" in section "Authentication"'
            )
        return domain

    @cached_property
    def issuer(self) -> str:
        return f"https://{self.domain}/"

    @cached_property
    def well_known_jwks(self) -> str:
        return f"https://{self.domain}/.well-known/jwks.json"

    @cached_property
    def jwks(self) -> Dict[str, Any]:
        response = requests.get(self.well_known_jwks)
        return json.loads(response.content)

    @cached_property
    def audience(self) -> str:
        authentication = self.authentication
        assert isinstance(authentication, dict)
        audience = authentication.get('Audience')
        if not audience:
            raise ServiceConfigError(
                'Missing key "Audience" in section "Authentication"'
            )
        return audience

    @cached_property
    def algorithms(self) -> List[str]:
        authentication = self.authentication
        assert isinstance(authentication, dict)
        algorithms = authentication.get('Algorithms', ["RS256"])
        if not algorithms:
            raise ServiceConfigError(
                'Value for key "Algorithms" in section'
                ' "Authentication" must not be empty'
            )
        return algorithms

    @cached_property
    def is_required(self) -> bool:
        authentication = self.authentication
        if not authentication:
            return False
        return authentication.get('IsRequired', False)

    @cached_property
    def granted_scopes(self) -> Optional[Set[str]]:
        # noinspection PyUnresolvedReferences
        id_token = self.get_id_token(
            require_auth=self.service_context.must_authenticate
        )
        return id_token.get('permissions') if id_token else None

    def get_id_token(self, require_auth: bool = False) \
            -> Optional[Mapping[str, str]]:
        """
        Decodes the access token is valid.
        """
        access_token = self.get_access_token(require_auth=require_auth)
        if access_token is None:
            return None

        if not self.authentication:
            if require_auth:
                raise ServiceAuthError(
                    "Invalid header",
                    log_message="Received access token,"
                                " but this server doesn't support"
                                " authentication."
                )
            return None

        try:
            unverified_header = jwt.get_unverified_header(access_token)
        except jwt.InvalidTokenError:
            raise ServiceAuthError(
                "Invalid header",
                log_message="Invalid header."
                            " Use an RS256 signed JWT Access Token"
            )
        if unverified_header["alg"] != "RS256":  # e.g. "HS256"
            raise ServiceAuthError(
                "Invalid header",
                log_message="Invalid header."
                            " Use an RS256 signed JWT Access Token"
            )

        jwks = self.jwks

        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break

        if rsa_key:
            try:
                id_token = jwt.decode(
                    access_token,
                    # TODO: this is stupid: we convert rsa_key to
                    #  JWT JSON only to produce the public key JSON string
                    jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(rsa_key)),
                    algorithms=self.algorithms,
                    audience=self.audience,
                    issuer=self.issuer
                )
            except jwt.ExpiredSignatureError:
                raise ServiceAuthError(
                    "Token expired",
                    log_message="Token is expired"
                )
            except jwt.InvalidTokenError:
                raise ServiceAuthError(
                    "Invalid claims",
                    log_message="Incorrect claims,"
                                " please check the audience and issuer"
                )
            except Exception:
                raise ServiceAuthError(
                    "Invalid header",
                    log_message="Unable to parse authentication token."
                )
            return id_token

        raise ServiceAuthError("Invalid header",
                               log_message="Unable to find appropriate key")

    def get_access_token(self, require_auth: bool = False) -> Optional[str]:
        """Obtains the access token from the Authorization Header
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
