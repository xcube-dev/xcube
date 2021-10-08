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

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Sequence, TypeVar, Generic, Type

from xcube.core.store import DatasetDescriptor
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

STATUS_IDS = ['ok', 'error', 'warning']

R = TypeVar('R', bound=JsonObject)


class GenericCubeGeneratorResult(Generic[R], JsonObject):
    def __init__(self,
                 status: str,
                 status_code: Optional[int] = None,
                 result: Optional[R] = None,
                 message: Optional[str] = None,
                 output: Optional[Sequence[str]] = None,
                 traceback: Optional[Sequence[str]] = None,
                 versions: Optional[Dict[str, str]] = None):
        assert_instance(status, str, name='status')
        assert_in(status, STATUS_IDS, name='status')
        self.status = status
        self.status_code = status_code
        self.result = result
        self.message = message if message else None
        self.output = list(output) if output else None
        self.traceback = list(traceback) if traceback else None
        self.versions = dict(versions) if versions else None

    def derive(self,
               /,
               status: Optional[str] = None,
               status_code: Optional[int] = None,
               result: Optional[R] = None,
               message: Optional[str] = None,
               output: Optional[Sequence[str]] = None,
               traceback: Optional[Sequence[str]] = None,
               versions: Optional[Dict[str, str]] = None) -> R:
        return self.__class__(status or self.status,
                              status_code=status_code or self.status_code,
                              result=result or self.result,
                              message=message or self.message,
                              output=output or self.output,
                              traceback=traceback or self.traceback,
                              versions=versions or self.versions)

    @classmethod
    @abstractmethod
    def get_result_schema(cls) -> JsonObjectSchema:
        """Get the JSON schema of the result object"""

    @classmethod
    def from_dict(cls, value: Dict) -> R:
        return cls.get_schema().from_instance(value)

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                status=JsonStringSchema(enum=STATUS_IDS),
                status_code=JsonIntegerSchema(),
                result=cls.get_result_schema(),
                message=JsonStringSchema(),
                output=JsonArraySchema(items=JsonStringSchema()),
                traceback=JsonArraySchema(items=JsonStringSchema()),
                versions=JsonObjectSchema(additional_properties=True)
            ),
            required=['status'],
            additional_properties=True,
            factory=cls,
        )


def make_cube_generator_result_class(result_type: Type[R]) \
        -> Type[GenericCubeGeneratorResult[R]]:
    class SpecificCubeGeneratorResult(GenericCubeGeneratorResult[R]):
        @classmethod
        def get_result_schema(cls) -> JsonObjectSchema:
            return result_type.get_schema()

    return SpecificCubeGeneratorResult


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
            factory=cls,
        )


CubeGeneratorResult = make_cube_generator_result_class(CubeReference)


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


CubeInfoResult = make_cube_generator_result_class(CubeInfo)
