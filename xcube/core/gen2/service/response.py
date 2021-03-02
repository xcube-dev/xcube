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

from typing import Optional, Dict, Union, List, Any

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from ..config import _to_dict
from ..response import CubeInfo
from ..response import ResponseBase


class Token(ResponseBase):
    def __init__(self, access_token: str, token_type: str):
        self.access_token = access_token
        self.token_type = token_type

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(access_token=JsonStringSchema(min_length=1),
                                                token_type=JsonStringSchema(min_length=1)),
                                required=['access_token', 'token_type'],
                                additional_properties=False,
                                factory=cls)

    @classmethod
    def from_dict(cls, value: Dict) -> 'Token':
        return cls.get_schema().from_instance(value)


class Status(ResponseBase):
    # noinspection PyUnusedLocal
    def __init__(self,
                 succeeded: int = None,
                 failed: int = None,
                 active: int = None,
                 start_time: str = None,
                 completion_time: str = None,
                 conditions: List[Dict[str, Any]] = None,
                 **additional_properties):
        self.succeeded: Optional[int] = succeeded
        self.failed: Optional[int] = failed
        self.active: Optional[int] = active
        self.start_time: Optional[str] = start_time
        self.completion_time: Optional[str] = completion_time
        self.conditions: Optional[Dict[str, Any]] = conditions
        self.additional_properties: Dict[str, Any] = additional_properties

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(succeeded=JsonIntegerSchema(nullable=True),
                                                failed=JsonIntegerSchema(nullable=True),
                                                active=JsonIntegerSchema(nullable=True),
                                                start_time=JsonStringSchema(nullable=True),
                                                completion_time=JsonStringSchema(nullable=True),
                                                conditions=JsonArraySchema(
                                                    items=JsonObjectSchema(additional_properties=True),
                                                    nullable=True
                                                )),
                                additional_properties=True,
                                factory=cls)

    @classmethod
    def from_dict(cls, value: Dict) -> 'Status':
        return cls.get_schema().from_instance(value)


class ProgressState(ResponseBase):
    def __init__(self,
                 progress: float,
                 # worked: Union[int, float],
                 total_work: Union[int, float],
                 **additional_properties):
        self.progress: float = float(progress)
        # self.worked: float = float(worked)
        self.total_work: float = float(total_work)
        self.additional_properties = additional_properties

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                progress=JsonNumberSchema(minimum=0.0),
                # worked=JsonNumberSchema(minimum=0),
                total_work=JsonNumberSchema(exclusive_minimum=0)
            ),
            required=[
                'progress',
                # 'worked',
                'total_work',
            ],
            additional_properties=True,
            factory=cls)

    @classmethod
    def from_dict(cls, value: Dict) -> 'ProgressState':
        return cls.get_schema().from_instance(value)


class Progress(ResponseBase):
    def __init__(self,
                 sender: str,
                 state: ProgressState):
        self.sender: str = sender
        self.state: ProgressState = state

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                sender=JsonStringSchema(),
                state=ProgressState.get_schema(),
            ),
            required=[
                'sender',
                'state',
            ],
            additional_properties=False,
            factory=cls)

    @classmethod
    def from_dict(cls, value: Dict) -> 'Progress':
        return cls.get_schema().from_instance(value)


class Result(ResponseBase):
    def __init__(self,
                 cubegen_id: str,
                 status: Status,
                 output: List[str] = None,
                 progress: List[Progress] = None,
                 **additional_properties):
        self.cubegen_id: str = cubegen_id
        self.status: Status = status
        self.output: Optional[List[str]] = output
        self.progress: Optional[List[Progress]] = progress
        self.additional_properties: Dict[str, Any] = additional_properties

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(cubegen_id=JsonStringSchema(min_length=1),
                                                status=Status.get_schema(),
                                                output=JsonArraySchema(
                                                    items=JsonStringSchema(),
                                                    nullable=True
                                                ),
                                                progress=JsonArraySchema(
                                                    items=Progress.get_schema(),
                                                    nullable=True
                                                )),
                                required=['cubegen_id', 'status'],
                                additional_properties=True,
                                factory=cls)

    @classmethod
    def from_dict(cls, value: Dict) -> 'Result':
        return cls.get_schema().from_instance(value)


class Response(ResponseBase):
    def __init__(self,
                 result: Result,
                 traceback: str = None,
                 **additional_properties):
        self.result: Result = result
        self.traceback: Optional[str] = traceback
        self.additional_properties = additional_properties

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(result=Result.get_schema(),
                                                traceback=JsonStringSchema()),
                                required=['result'],
                                additional_properties=True,
                                factory=cls)

    @classmethod
    def from_dict(cls, value: Dict) -> 'Response':
        return cls.get_schema().from_instance(value)


class CostInfo(ResponseBase):
    # def __init__(self,
    #              punits_input: int,
    #              punits_output: int,
    #              punits_combined: int):
    #     self.punits_input: int = punits_input
    #     self.punits_output: int = punits_output
    #     self.punits_combined: int = punits_combined
    def __init__(self,
                 **additional_properties):
        self.additional_properties = additional_properties

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        # return JsonObjectSchema(properties=dict(punits_input=JsonIntegerSchema(),
        #                                         punits_output=JsonIntegerSchema(),
        #                                         punits_combined=JsonIntegerSchema()),
        #                         required=['punits_input', 'punits_output', 'punits_combined'],
        #                         additional_properties=True,
        #                         factory=cls)
        return JsonObjectSchema(additional_properties=True,
                                factory=cls)

    @classmethod
    def from_dict(cls, value: Dict) -> 'CostInfo':
        return cls.get_schema().from_instance(value)


class CubeInfoWithCosts(CubeInfo):
    def __init__(self, cost_info: CostInfo, **kwargs):
        super().__init__(**kwargs)
        self.cost_info: CostInfo = cost_info

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        schema = super().get_schema()
        schema.properties.update(cost_info=CostInfo.get_schema())
        schema.required.add('cost_info')
        return schema

    @classmethod
    def from_dict(cls, value: Dict) -> 'CubeInfoWithCosts':
        return cls.get_schema().from_instance(value)

    def to_dict(self) -> Dict:
        """Convert this instance to a dictionary."""
        d = _to_dict(self, tuple(CubeInfo.get_schema().properties.keys()))
        cost_info = _to_dict(self.cost_info.additional_properties, tuple(CostInfo.get_schema().properties.keys()))
        d.update(cost_info=cost_info)
        return d
