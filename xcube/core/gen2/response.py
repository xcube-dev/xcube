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

from abc import ABC, abstractmethod
from typing import Dict, Any

from xcube.util.jsonschema import JsonObjectSchema
from .config import _to_dict


class ResponseBase(ABC):

    @classmethod
    @abstractmethod
    def get_schema(cls) -> JsonObjectSchema:
        """Get JSON object schema."""

    @classmethod
    def from_dict(cls, value: Dict) -> 'ResponseBase':
        """Create instance from dictionary *value*."""
        return cls.get_schema().from_instance(value)


class CubeInfo(ResponseBase):
    def __init__(self,
                 dims: Dict[str, int],
                 chunks: Dict[str, int],
                 data_vars: Dict[str, Dict[str, Any]]):
        self.dims = dims
        self.chunks = chunks
        self.data_vars = data_vars

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(dims=JsonObjectSchema(additional_properties=True),
                                                chunks=JsonObjectSchema(additional_properties=True),
                                                data_vars=JsonObjectSchema(additional_properties=True)),
                                required=['dims', 'chunks', 'data_vars'],
                                additional_properties=False,
                                factory=cls)

    @classmethod
    def from_dict(cls, value: Dict) -> 'CubeInfo':
        return cls.get_schema().from_instance(value)

    def to_dict(self) -> Dict:
        """Convert this instance to a dictionary."""
        return _to_dict(self, tuple(self.get_schema().properties.keys()))
