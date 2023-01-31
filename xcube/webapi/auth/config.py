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

from typing import List, Optional, Any, Mapping

from xcube.constants import LOG
from xcube.server.api import ApiError
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

AUTHENTICATION_SCHEMA = JsonObjectSchema(
    properties=dict(
        Authority=JsonStringSchema(format='uri'),
        Domain=JsonStringSchema(description='Deprecated, use Authority'),
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
    additional_properties=False
)


class AuthConfig:
    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                Authentication=AUTHENTICATION_SCHEMA
            ),
            factory=AuthConfig.from_dict
        )

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
    def from_config(cls, config: Mapping[str, Any]) -> Optional['AuthConfig']:
        authentication = config.get('Authentication')
        if authentication is None:
            return None
        return cls.from_dict(authentication)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Optional['AuthConfig']:
        authority = config.get('Authority')
        domain = config.get('Domain')
        if domain:
            LOG.warning('Configuration parameter "Domain"'
                        ' in section "Authentication"'
                        ' has been deprecated. Use "Authority" instead.')
            if not authority:
                authority = f"https://{domain}"
        if not authority:
            raise ApiError.InvalidServerConfig(
                'Missing key "Authority" in section "Authentication"'
            )
        audience = config.get('Audience')
        if not audience:
            raise ApiError.InvalidServerConfig(
                'Missing key "Audience" in section "Authentication"'
            )
        algorithms = config.get('Algorithms', ['RS256'])
        if not algorithms:
            raise ApiError.InvalidServerConfig(
                'Value for key "Algorithms" in section'
                ' "Authentication" must not be empty'
            )
        is_required = config.get('IsRequired', False)
        return AuthConfig(authority,
                          audience,
                          algorithms,
                          is_required=is_required)
