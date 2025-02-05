# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Dict, List, Optional, Union

from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonIntegerSchema,
    JsonNumberSchema,
    JsonObject,
    JsonObjectSchema,
    JsonStringSchema,
)

from ..response import CubeGeneratorResult, CubeInfo, make_cube_generator_result_class


class CubeGeneratorToken(JsonObject):
    def __init__(self, access_token: str, token_type: str):
        self.access_token = access_token
        self.token_type = token_type

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                access_token=JsonStringSchema(min_length=1),
                token_type=JsonStringSchema(min_length=1),
            ),
            required=["access_token", "token_type"],
            additional_properties=False,
            factory=cls,
        )

    @classmethod
    def from_dict(cls, value: dict) -> "CubeGeneratorToken":
        return cls.get_schema().from_instance(value)


class CubeGeneratorJobStatus(JsonObject):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        succeeded: int = None,
        failed: int = None,
        active: int = None,
        start_time: str = None,
        completion_time: str = None,
        conditions: list[dict[str, Any]] = None,
        **additional_properties,
    ):
        self.succeeded: Optional[int] = succeeded
        self.failed: Optional[int] = failed
        self.active: Optional[int] = active
        self.start_time: Optional[str] = start_time
        self.completion_time: Optional[str] = completion_time
        self.conditions: Optional[dict[str, Any]] = conditions
        self.additional_properties: dict[str, Any] = additional_properties

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                succeeded=JsonIntegerSchema(nullable=True),
                failed=JsonIntegerSchema(nullable=True),
                active=JsonIntegerSchema(nullable=True),
                start_time=JsonStringSchema(nullable=True),
                completion_time=JsonStringSchema(nullable=True),
                conditions=JsonArraySchema(
                    items=JsonObjectSchema(additional_properties=True), nullable=True
                ),
            ),
            additional_properties=True,
            factory=cls,
        )

    @classmethod
    def from_dict(cls, value: dict) -> "CubeGeneratorJobStatus":
        return cls.get_schema().from_instance(value)


class CubeGeneratorProgressState(JsonObject):
    """Current progress state of the remote generator."""

    def __init__(
        self,
        progress: float,
        # worked: Union[int, float],
        total_work: Union[int, float],
        **additional_properties,
    ):
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
                total_work=JsonNumberSchema(exclusive_minimum=0),
            ),
            required=[
                "progress",
                # 'worked',
                "total_work",
            ],
            additional_properties=True,
            factory=cls,
        )

    @classmethod
    def from_dict(cls, value: dict) -> "CubeGeneratorProgressState":
        return cls.get_schema().from_instance(value)


class CubeGeneratorProgress(JsonObject):
    """Current progress of the remote generator."""

    def __init__(self, sender: str, state: CubeGeneratorProgressState):
        self.sender: str = sender
        self.state: CubeGeneratorProgressState = state

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                sender=JsonStringSchema(),
                state=CubeGeneratorProgressState.get_schema(),
            ),
            required=[
                "sender",
                "state",
            ],
            additional_properties=False,
            factory=cls,
        )

    @classmethod
    def from_dict(cls, value: dict) -> "CubeGeneratorProgress":
        return cls.get_schema().from_instance(value)


class CubeGeneratorState(JsonObject):
    """Current state of the remote generator."""

    def __init__(
        self,
        job_id: str,
        job_status: CubeGeneratorJobStatus,
        job_result: Optional[CubeGeneratorResult] = None,
        output: Optional[list[str]] = None,
        progress: Optional[list[CubeGeneratorProgress]] = None,
        **additional_properties,
    ):
        self.job_id = job_id
        self.job_status = job_status
        self.job_result = job_result
        self.output = output
        self.progress = progress
        self.additional_properties = additional_properties

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                job_id=JsonStringSchema(min_length=1),
                job_status=CubeGeneratorJobStatus.get_schema(),
                job_result=CubeGeneratorResult.get_schema(),
                output=JsonArraySchema(items=JsonStringSchema(), nullable=True),
                progress=JsonArraySchema(
                    items=CubeGeneratorProgress.get_schema(), nullable=True
                ),
            ),
            required=["job_id", "job_status"],
            additional_properties=True,
            factory=cls,
        )

    @classmethod
    def from_dict(cls, value: dict) -> "CubeGeneratorState":
        return cls.get_schema().from_instance(value)


class CostEstimation(JsonObject):
    def __init__(self, required: int, available: int = None, limit: int = None):
        self.available: int = available
        self.limit: Optional[int] = limit
        self.required: Optional[int] = required

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                required=JsonIntegerSchema(minimum=0),
                available=JsonIntegerSchema(minimum=0),
                limit=JsonIntegerSchema(minimum=0),
            ),
            required=["required"],
            additional_properties=False,
            factory=cls,
        )

    @classmethod
    def from_dict(cls, value: dict) -> "CostEstimation":
        return cls.get_schema().from_instance(value)


class CubeInfoWithCosts(CubeInfo):
    def __init__(self, cost_estimation: CostEstimation, **kwargs):
        super().__init__(**kwargs)
        self.cost_estimation: CostEstimation = cost_estimation

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        schema = super().get_schema()
        schema.properties.update(cost_estimation=CostEstimation.get_schema())
        schema.required.append("cost_estimation")
        schema.factory = cls
        return schema


CubeInfoWithCostsResult = make_cube_generator_result_class(CubeInfoWithCosts)
