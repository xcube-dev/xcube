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
import abc
from typing import Any, Dict

from xcube.util.jsonschema import JsonSchema, JsonObjectSchema, JsonIntegerSchema, JsonStringSchema

DEFAULT_PORT = 8080

DEFAULT_ADDRESS = "0.0.0.0"


class Config(abc.ABC):
    """Configuration interface."""

    @classmethod
    @abc.abstractmethod
    def get_schema(cls) -> JsonSchema:
        """Get the JSON Schema for the configuration."""

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create a new configuration from given JSON-serializable dictionary."""

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert this configuration into a JSON-serializable dictionary."""


class ServerConfig(Config):
    """A server configuration."""

    def __init__(self, address: str, port: int):
        self.address = address
        self.port = port

    @classmethod
    def get_schema(cls) -> JsonSchema:
        return JsonObjectSchema(
            properties=dict(
                address=JsonStringSchema(default=DEFAULT_ADDRESS),
                port=JsonIntegerSchema(default=DEFAULT_PORT),
            ),
            required=["address", "port"]
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServerConfig":
        return ServerConfig(d.get("address", DEFAULT_ADDRESS),
                            d.get("port", DEFAULT_PORT))

    def to_dict(self):
        return dict(
            address=self.address,
            port=self.port
        )
