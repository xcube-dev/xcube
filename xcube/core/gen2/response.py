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
from abc import abstractmethod, ABC
from typing import Dict, Any, Optional, Sequence

from xcube.core.store import DatasetDescriptor
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

STATUS_IDS = ['ok', 'error', 'warning']


class AbstractResult(JsonObject, ABC):
    def __init__(self,
                 status: str,
                 result: Optional[Any] = None,
                 message: Optional[str] = None,
                 output: Optional[Sequence[str]] = None,
                 traceback: Optional[Sequence[str]] = None):
        assert_instance(status, str, name='status')
        assert_in(status, STATUS_IDS, name='status')
        self.result = result
        self.status = status
        self.message = message if message else None
        self.output = list(output) if output else None
        self.traceback = list(traceback) if traceback else None

    @classmethod
    @abstractmethod
    def get_result_schema(cls) -> JsonObjectSchema:
        """Get the JSON schema of the result object"""

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                status=JsonStringSchema(enum=STATUS_IDS),
                result=cls.get_result_schema(),
                message=JsonStringSchema(),
                output=JsonArraySchema(items=JsonStringSchema()),
                traceback=JsonArraySchema(items=JsonStringSchema()),
            ),
            required=['status'],
            additional_properties=True,
            factory=cls,
        )


class CubeReference(JsonObject):
    def __init__(self, data_id: str):
        self.data_id = data_id

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                data_id=JsonStringSchema(min_length=1),
            ),
            required=['data_id'],
            additional_properties=False,
        )


class CubeGeneratorResult(AbstractResult):
    def __init__(self,
                 status: str,
                 result: Optional[CubeReference] = None,
                 message: Optional[str] = None,
                 output: Optional[Sequence[str]] = None,
                 traceback: Optional[Sequence[str]] = None):
        super().__init__(status,
                         result=result,
                         message=message,
                         output=output,
                         traceback=traceback)

    @classmethod
    def get_result_schema(cls) -> JsonObjectSchema:
        return CubeReference.get_schema()

    @classmethod
    def from_dict(cls, value: Dict) -> 'CubeGeneratorResult':
        return cls.get_schema().from_instance(value)


class CubeInfo(JsonObject):
    def __init__(self,
                 dataset_descriptor: DatasetDescriptor,
                 size_estimation: Dict[str, Any],
                 **kwargs):
        self.dataset_descriptor: DatasetDescriptor = dataset_descriptor
        self.size_estimation: dict = size_estimation

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                dataset_descriptor=DatasetDescriptor.get_schema(),
                size_estimation=JsonObjectSchema(additional_properties=True)
            ),
            required=['dataset_descriptor', 'size_estimation'],
            additional_properties=True,
            factory=cls
        )


class CubeInfoResult(AbstractResult):
    def __init__(self,
                 status: str,
                 result: Optional[CubeInfo] = None,
                 message: Optional[str] = None,
                 output: Optional[Sequence[str]] = None,
                 traceback: Optional[Sequence[str]] = None):
        super().__init__(status,
                         result=result,
                         message=message,
                         output=output,
                         traceback=traceback)

    @classmethod
    def get_result_schema(cls) -> JsonObjectSchema:
        return CubeInfo.get_schema()

    @classmethod
    def from_dict(cls, value: Dict) -> 'CubeInfoResult':
        return cls.get_schema().from_instance(value)
