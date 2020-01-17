"""Python Flask API Auth0 integration example
"""

import json
from typing import Optional, Mapping, List, Dict, Any

import jwt
import requests
from jwt.algorithms import RSAAlgorithm

from xcube.webapi.errors import ServiceAuthError, ServiceConfigError


class AuthConfig:
    def __init__(self, domain: str, audience: str, algorithms: List[str]):
        self._domain = domain
        self._audience = audience
        self._algorithms = algorithms

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def issuer(self) -> str:
        return f"https://{self.domain}/"

    @property
    def well_known_jwks(self) -> str:
        return f"https://{self.domain}/.well-known/jwks.json"

    @property
    def audience(self) -> str:
        return self._audience

    @property
    def algorithms(self) -> List[str]:
        return self._algorithms

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional['AuthConfig']:
        authentication = config.get('Authentication')
        if not authentication:
            return None
        domain = authentication.get('Domain')
        if not domain:
            raise ServiceConfigError('Missing key "Domain" in section "Authentication"')
        audience = authentication.get('Audience')
        if not audience:
            raise ServiceConfigError('Missing key "Audience" in section "Authentication"')
        algorithms = authentication.get('Algorithms', ['RS256'])
        if not algorithms:
            raise ServiceConfigError('Missing key "Algorithms" in section "Authentication"')
        return AuthConfig(domain, audience, algorithms)


class AuthMixin:

    @property
    def auth_config(self) -> Optional[AuthConfig]:
        # noinspection PyUnresolvedReferences
        return AuthConfig.from_config(self.service_context.config)

    def authorize_for(self, *required_permissions: List[str], fail_fast: bool = False) -> bool:
        id_token = self.get_id_token(require=True)
        granted_permissions = set(id_token.get('permissions', []))
        if not granted_permissions:
            raise ServiceAuthError('Token without permissions', log_message='Token is missing permissions')
        for required_permission in required_permissions:
            if required_permission not in granted_permissions:
                if fail_fast:
                    raise ServiceAuthError('Missing permission', log_message='Permissions are not sufficient')
                return False
        return True

    def get_access_token(self, require: bool = False) -> Optional[str]:
        """Obtains the access token from the Authorization Header
        """
        # noinspection PyUnresolvedReferences
        auth = self.request.headers.get("Authorization", None)
        if not auth:
            if require:
                raise ServiceAuthError("Authorization header missing",
                                       log_message="Authorization header is expected")
            return None

        parts = auth.split()

        if parts[0].lower() != "bearer":
            raise ServiceAuthError("Invalid header",
                                   log_message="Authorization header must start with Bearer")
        elif len(parts) == 1:
            raise ServiceAuthError("Invalid header",
                                   log_message="Token not found")
        elif len(parts) > 2:
            raise ServiceAuthError("Invalid header",
                                   log_message="Authorization header must be Bearer token")

        token = parts[1]
        return token

    def get_id_token(self, require: bool = False) -> Optional[Mapping[str, str]]:
        """
        Decodes the access token is valid.
        """
        access_token = self.get_access_token(require=require)
        if access_token is None:
            return None

        auth_config = self.auth_config
        if auth_config is None:
            if require:
                raise ServiceAuthError("Invalid header",
                                       log_message="Received access token, "
                                                   "but this server doesn't support authentication.")
            return None

        try:
            unverified_header = jwt.get_unverified_header(access_token)
        except jwt.InvalidTokenError:
            raise ServiceAuthError("Invalid header",
                                   log_message="Invalid header. Use an RS256 signed JWT Access Token")
        if unverified_header["alg"] == "HS256":
            raise ServiceAuthError("Invalid header",
                                   log_message="Invalid header. Use an RS256 signed JWT Access Token")

        # TODO: read jwks from cache
        response = requests.get(auth_config.well_known_jwks)
        jwks = json.loads(response.content)

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
                    # TODO: this is stupid: we convert rsa_key to JWT JSON only to produce the public key
                    RSAAlgorithm.from_jwk(json.dumps(rsa_key)),
                    algorithms=auth_config.algorithms,
                    audience=auth_config.audience,
                    issuer=auth_config.issuer
                )
            except jwt.ExpiredSignatureError:
                raise ServiceAuthError("Token expired",
                                       log_message="Token is expired")
            except jwt.InvalidTokenError:
                raise ServiceAuthError("Invalid claims",
                                       log_message="Incorrect claims, please check the audience and issuer")
            except Exception:
                raise ServiceAuthError("Invalid header",
                                       log_message="Unable to parse authentication token.")
            return id_token

        raise ServiceAuthError("Invalid header",
                               log_message="Unable to find appropriate key")
