# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import List, Optional, Any
from collections.abc import Mapping

from xcube.constants import LOG
from xcube.server.api import ApiError
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

AUTHENTICATION_SCHEMA = JsonObjectSchema(
    properties=dict(
        Authority=JsonStringSchema(format="uri"),
        Domain=JsonStringSchema(description="Deprecated, use Authority"),
        Audience=JsonStringSchema(),
        Algorithms=JsonArraySchema(
            items=JsonStringSchema(default=["RS256"]),
        ),
        IsRequired=JsonBooleanSchema(default=False),
    ),
    required=[
        # Either "Authority" or "Domain" are required
        # "Authority",
        "Audience"
    ],
    additional_properties=False,
)


class AuthConfig:
    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(Authentication=AUTHENTICATION_SCHEMA),
            factory=AuthConfig.from_dict,
        )

    def __init__(
        self,
        authority: str,
        audience: str,
        algorithms: list[str],
        is_required: bool = False,
    ):
        self._authority = authority
        self._audience = audience
        self._algorithms = algorithms
        self._is_required = is_required

    @property
    def authority(self) -> str:
        return self._authority

    @property
    def _norm_authority(self) -> str:
        return self.authority[:-1] if self.authority.endswith("/") else self.authority

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
    def algorithms(self) -> list[str]:
        return self._algorithms

    @property
    def is_required(self) -> bool:
        return self._is_required

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Optional["AuthConfig"]:
        authentication = config.get("Authentication")
        if authentication is None:
            return None
        return cls.from_dict(authentication)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Optional["AuthConfig"]:
        authority = config.get("Authority")
        domain = config.get("Domain")
        if domain:
            LOG.warning(
                'Configuration parameter "Domain"'
                ' in section "Authentication"'
                ' has been deprecated. Use "Authority" instead.'
            )
            if not authority:
                authority = f"https://{domain}"
        if not authority:
            raise ApiError.InvalidServerConfig(
                'Missing key "Authority" in section "Authentication"'
            )
        audience = config.get("Audience")
        if not audience:
            raise ApiError.InvalidServerConfig(
                'Missing key "Audience" in section "Authentication"'
            )
        algorithms = config.get("Algorithms", ["RS256"])
        if not algorithms:
            raise ApiError.InvalidServerConfig(
                'Value for key "Algorithms" in section'
                ' "Authentication" must not be empty'
            )
        is_required = config.get("IsRequired", False)
        return AuthConfig(authority, audience, algorithms, is_required=is_required)
