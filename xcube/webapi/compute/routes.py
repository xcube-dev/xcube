# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import jsonschema

from xcube.server.api import ApiError, ApiHandler

from .api import api
from .context import JOB_STATUSES, ComputeContext
from .controllers import get_compute_operation, get_compute_operations

OP_SCHEMA = {
    "type": "object",
    "properties": {
        "operationId": {"type": "string"},
        "parametersSchema": {
            "type": "array",
            "items": {
                "type": "object",
                "description": "JSON Schema for each parameter",
            },
        },
        "description": {"type": "string"},
    },
    "required": ["operationId", "parametersSchema"],
}

OP_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {"type": "array", "items": OP_SCHEMA},
    },
    "required": ["operations"],
}

OP_RESPONSES_SCHEMA = {
    "200": {
        "description": "Compute operation details",
        "content": {"application/json": OP_SCHEMA},
    },
    "404": {"description": "An operation with the specified ID was not found."},
}

OP_LIST_RESPONSES_SCHEMA = {
    "200": {
        "description": "Compute operation details",
        "content": {"application/json": {"schema": OP_LIST_SCHEMA}},
    },
    "404": {"description": "An operation with the specified ID was not found."},
}

JOB_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(JOB_STATUSES)},
        "error": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "type": {"type": "string"},
                "traceback": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["message"],
        },
    },
}

JOB_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "datasetId": {"type": "string"},
        "data": {},
    },
}

JOB_SCHEMA = {
    "type": "object",
    "properties": {
        "jobId": {"type": "integer"},
        "state": JOB_STATE_SCHEMA,
        "result": JOB_RESULT_SCHEMA,
    },
    "required": ["jobId", "state"],
}

JOB_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "jobId": {"type": "integer"},
                    "operationId": {"type": "string"},
                    "status": {"type": "string", "enum": list(JOB_STATUSES)},
                },
                "required": ["jobId", "operationId", "status"],
            },
        },
    },
    "required": ["jobs"],
}

JOB_RESPONSES_SCHEMA = {
    "200": {
        "description": "Compute job details",
        "content": {"application/json": JOB_SCHEMA},
    },
    "404": {"description": "A job with the specified ID was not found."},
}

JOB_LIST_RESPONSES_SCHEMA = {
    "200": {
        "description": "List of compute jobs",
        "content": {"application/json": {"schema": JOB_LIST_SCHEMA}},
    }
}

JOB_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "operationId": {"type": "string"},
        "parameters": {"type": "object", "items": {}},  # Any
        "output": {
            "type": "object",
            "properties": {
                "datasetId": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "style": {"type": "string"},
                "colorMappings": {"type": "object"},
            },
            # Doesn't work with swagger:
            # TypeError: t.get is not a function
            # "additionalProperties": False,
        },
        # Doesn't work with swagger:
        # TypeError: t.get is not a function
        # "additionalProperties": False,
    },
    "required": ["operationId"],
    "example": {
        "operationId": "spatial_subset",
        "parameters": {"dataset": "demo", "bbox": [1, 51, 4, 52]},
        "output": {"datasetId": "demo_subset", "title": "My demo subset"},
    },
}

JOB_REQUEST_BODY_SCHEMA = {
    "description": "Compute job request",
    "required": True,
    "content": {"application/json": {"schema": JOB_REQUEST_SCHEMA}},
}

PATH_PARAM_OP_ID = {
    "name": "operationId",
    "in": "path",
    "description": "Operation identifier",
    "schema": {"type": "string"},
}

PATH_PARAM_JOB_ID = {
    "name": "jobId",
    "in": "path",
    "description": "Job identifier",
    "schema": {"type": "integer"},
}


@api.route("/compute/operations")
class ComputeOperationsHandler(ApiHandler[ComputeContext]):
    @api.operation(
        operation_id="getComputeOperations",
        summary="Get available compute operations.",
        responses=OP_LIST_RESPONSES_SCHEMA,
    )
    def get(self):
        self.response.finish(get_compute_operations(self.ctx))


# noinspection PyPep8Naming
@api.route("/compute/operations/{operationId}")
class ComputeOperationHandler(ApiHandler[ComputeContext]):
    @api.operation(
        operation_id="getComputeOperationInfo",
        summary="Get the details of a given compute operation.",
        parameters=[PATH_PARAM_OP_ID],
        responses=OP_RESPONSES_SCHEMA,
    )
    def get(self, operationId):
        self.response.finish(get_compute_operation(self.ctx, operationId))


# noinspection PyPep8Naming
@api.route("/compute/jobs")
class ComputeJobsHandler(ApiHandler[ComputeContext]):
    @api.operation(
        operation_id="getComputeJobs",
        summary="Get recent compute jobs.",
        responses=JOB_LIST_RESPONSES_SCHEMA,
    )
    def get(self):
        self.response.finish(
            {
                "jobs": [
                    {
                        "jobId": job["jobId"],
                        "operationId": job["request"]["operationId"],
                        "status": job["state"]["status"],
                    }
                    for job in self.ctx.jobs.values()
                ]
            }
        )

    @api.operation(
        operation_id="scheduleComputeJob",
        summary="Schedule a new compute job.",
        request_body=JOB_REQUEST_BODY_SCHEMA,
        responses=JOB_RESPONSES_SCHEMA,
    )
    def put(self):
        # TODO: build a more complete schema and publish it as OpenAPI?
        #  Currently, the schema is built for the current request and only
        #  contains (and only needs to contain) a parameter schema for the
        #  requested operation. In xcube’s live OpenAPI specification,
        #  the parameters are just an object with no further constraints.
        #  Ideally we’d want to detail the parameters for the various
        #  available operations in the OpenAPI specification. This can be
        #  done with if-then constructs: see
        #  https://json-schema.org/understanding-json-schema/reference/conditionals.html#if-then-else
        job_request = self.request.json
        basic_schema = self.put.__openapi__["requestBody"]["content"][
            "application/json"
        ]["schema"]
        # noinspection PyProtectedMember,PyUnresolvedReferences
        op_schema = dict(
            properties=dict(
                operationId=dict(enum=list(self.ctx.op_registry.ops.keys())),
                parameters=self.ctx.op_registry.ops[
                    job_request["operationId"]
                ]._op_info.params_schema,
            )
        )
        full_schema = {**basic_schema, **op_schema}
        try:
            jsonschema.validate(job_request, full_schema)
            self.response.finish(self.ctx.schedule_job(job_request))
        except jsonschema.ValidationError as e:
            raise ApiError.BadRequest(message=f"{e.message} at {e.json_path}")


# noinspection PyPep8Naming
@api.route("/compute/jobs/{jobId}")
class ComputeJobHandler(ApiHandler[ComputeContext]):
    @api.operation(
        operation_id="getComputeJob",
        summary="Get details about a compute job.",
        parameters=[PATH_PARAM_JOB_ID],
        responses=JOB_RESPONSES_SCHEMA,
    )
    def get(self, jobId: str):
        job = self.ctx.jobs.get(int(jobId))
        if job is None:
            raise ApiError.NotFound(f"job #{jobId} cannot be found")
        self.response.finish(job)

    @api.operation(
        operation_id="cancelComputeJob",
        summary="Cancel an existing compute job.",
        parameters=[PATH_PARAM_JOB_ID],
        responses=JOB_RESPONSES_SCHEMA,
    )
    def delete(self, jobId: str):
        self.response.finish(self.ctx.cancel_job(int(jobId)))
