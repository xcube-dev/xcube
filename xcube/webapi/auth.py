"""Python Flask API Auth0 integration example
"""

import json
from typing import Optional, Mapping, List, Dict, Any, Set

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
            raise ServiceConfigError('Value for key "Algorithms" in section "Authentication" must not be empty')
        return AuthConfig(domain, audience, algorithms)


class AuthMixin:

    @property
    def auth_config(self) -> Optional[AuthConfig]:
        # noinspection PyUnresolvedReferences
        return AuthConfig.from_config(self.service_context.config)

    @property
    def granted_scopes(self) -> Set[str]:
        id_token = self.get_id_token(require_auth=False)
        if not id_token:
            return set()
        return set(id_token.get('permissions', []))

    def get_access_token(self, require_auth: bool = False) -> Optional[str]:
        """Obtains the access token from the Authorization Header
        """
        # noinspection PyUnresolvedReferences
        auth = self.request.headers.get("Authorization", None)
        if not auth:
            if require_auth:
                raise ServiceAuthError("Authorization header missing",
                                       log_message="Authorization header is expected")
            return None

        parts = auth.split()

        if parts[0].lower() != "bearer":
            raise ServiceAuthError("Invalid header",
                                   log_message='Authorization header must start with "Bearer"')
        elif len(parts) == 1:
            raise ServiceAuthError("Invalid header",
                                   log_message="Bearer token not found")
        elif len(parts) > 2:
            raise ServiceAuthError("Invalid header",
                                   log_message="Authorization header must be Bearer token")

        return parts[1]

    def get_id_token(self, require_auth: bool = False) -> Optional[Mapping[str, str]]:
        """
        Decodes the access token is valid.
        """
        access_token = self.get_access_token(require_auth=require_auth)
        if access_token is None:
            return None

        auth_config = self.auth_config
        if auth_config is None:
            if require_auth:
                raise ServiceAuthError("Invalid header",
                                       log_message="Received access token, "
                                                   "but this server doesn't support authentication.")
            return None

        try:
            unverified_header = jwt.get_unverified_header(access_token)
        except jwt.InvalidTokenError:
            raise ServiceAuthError("Invalid header",
                                   log_message="Invalid header. Use an RS256 signed JWT Access Token")
        if unverified_header["alg"] != "RS256":  # e.g. "HS256"
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
                    # TODO: this is stupid: we convert rsa_key to JWT JSON only to produce the public key JSON string
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
            # except Exception:
            #     raise ServiceAuthError("Invalid header",
            #                            log_message="Unable to parse authentication token.")
            return id_token

        raise ServiceAuthError("Invalid header",
                               log_message="Unable to find appropriate key")


def assert_scopes(required_scopes: Set[str],
                  granted_scopes: Set[str]):
    """
    Assert scopes.
    Raise ServiceAuthError if one of *required_scopes* is not in *granted_scopes*.

    :param required_scopes: The list of required scopes
    :param granted_scopes: The set of granted scopes
    """
    missing_scope = _get_missing_scope(required_scopes, granted_scopes)
    if missing_scope is not None:
        raise ServiceAuthError('Missing permission', log_message=f'Missing permission {missing_scope}')


def check_scopes(required_scopes: Set[str],
                 granted_scopes: Set[str],
                 is_substitute: bool = False) -> bool:
    """
    Check scopes.

    This function is used to filter out a resource's sub-resources for which a given client has no permission.

    If one of *required_scopes* is not in *granted_scopes*, fail.
    If *granted_scopes* exists and *is_substitute*, fail too.
    Else succeed.

    :param required_scopes: The list of required scopes
    :param granted_scopes: The set of granted scopes
    :param is_substitute: True, if the resource to be checked is a substitute.
    :return: True, if scopes are ok.
    """
    return _get_missing_scope(required_scopes, granted_scopes, is_substitute=is_substitute) is None


def _get_missing_scope(required_scopes: Set[str],
                       granted_scopes: Set[str],
                       is_substitute: bool = False) -> Optional[str]:
    for required_scope in required_scopes:
        if required_scope not in granted_scopes:
            # If the required scope is not a granted scope, fail
            return required_scope
    # If there are granted scopes then the client is authorized,
    # hence fail for substitute resources (e.g. demo resources) as there is usually
    # a better (non-demo) resource that replaces it.
    if granted_scopes and is_substitute:
        return '<is_substitute>'
    # All ok.
    return None
