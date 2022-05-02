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

from typing import Dict

from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonSchema
from xcube.util.jsonschema import JsonStringSchema

DEFAULT_PORT = 8080
DEFAULT_ADDRESS = "0.0.0.0"


class ServerConfig:
    """A server configuration."""

    def __init__(self, **properties):
        for k, v in properties.items():
            setattr(self, k, v)

    @classmethod
    def get_schema(cls, **api_schemas: Dict[str, JsonSchema]) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                port=JsonIntegerSchema(default=DEFAULT_PORT),
                address=JsonStringSchema(default=DEFAULT_ADDRESS),
                **api_schemas
            ),
            additional_properties=False,
            factory=cls
        )
