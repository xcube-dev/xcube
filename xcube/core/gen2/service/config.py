# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import json

import yaml

from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


class ServiceConfig:
    def __init__(self,
                 endpoint_url: str = None,
                 client_id: str = None,
                 client_secret: str = None,
                 access_token: str = None):
        # TODO: validate
        self.endpoint_url = endpoint_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token

    @classmethod
    def from_file(cls, service_config_file: str) -> 'ServiceConfig':
        with open(service_config_file, 'r') as fp:
            if service_config_file.endswith('.json'):
                service_config = json.load(fp)
            else:
                service_config = yaml.safe_load(fp)
        cls.get_schema().validate_instance(service_config)
        return ServiceConfig(**service_config)

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                endpoint_url=JsonStringSchema(min_length=1),
                client_id=JsonStringSchema(min_length=1),
                client_secret=JsonStringSchema(min_length=1),
                access_token=JsonObjectSchema(min_length=1),
            ),
            additional_properties=False,
            required=[],
            factory=cls,
        )

    def to_dict(self):
        from ..config import _to_dict
        return _to_dict(self, ('endpoint_url',
                               'client_id',
                               'client_secret',
                               'access_token'))
