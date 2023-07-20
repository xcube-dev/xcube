# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

import jsonschema
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from .api import api
from .context import ComputeContext
from .controllers import get_compute_operations
from .controllers import get_compute_operation
from .context import JOB_STATUSES

OP_SCHEMA = {
    "type": "object",
    "properties": {
        "operationId": {"type": "string"},
        "parametersSchema": {
            "type": "array",
            "items": {
                "type": "object",
                "description": "JSON Schema for each parameter",
            }
        },
        "description": {"type": "string"},
    },
    "required": ["operationId", "parametersSchema"],
}

OP_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": OP_SCHEMA
        },
    },
    "required": ["operations"],
}

OP_RESPONSES_SCHEMA = {
    "200": {
        "description": "Compute operation details",
        "content": {
            "application/json": OP_SCHEMA
        }
    },
    "404": {
        "description": "An operation with the specified ID was not found."
    }
}

OP_LIST_RESPONSES_SCHEMA = {
    "200": {
        "description": "Compute operation details",
        "content": {
            "application/json": {
                "schema": OP_LIST_SCHEMA
            }
        }
    },
    "404": {
        "description": "An operation with the specified ID was not found."
    }
}

JOB_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": list(JOB_STATUSES)
        },
        "error": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "type": {"type": "string"},
                "traceback": {
                    "type": "array",
                    "items": {"type": "string"}
                },
            },
            "required": ["message"]
        },
    }
}

JOB_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "datasetId": {"type": "string"},
        "data": {},
    }
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
                    "status": {
                        "type": "string",
                        "enum": list(JOB_STATUSES)
                    },
                },
                "required": ["jobId", "operationId", "status"],
            }
        },
    },
    "required": ["jobs"],
}

JOB_RESPONSES_SCHEMA = {
    "200": {
        "description": "Compute job details",
        "content": {
            "application/json": JOB_SCHEMA
        }
    },
    "404": {
        "description": "A job with the specified ID was not found."
    }
}

JOB_LIST_RESPONSES_SCHEMA = {
    "200": {
        "description": "List of compute jobs",
        "content": {
            "application/json": {
                "schema": JOB_LIST_SCHEMA
            }
        }
    }
}

JOB_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "operationId": {
            "type": "string"
        },
        "parameters": {
            "type": "object",
            "items": {}  # Any
        },
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
        "parameters": {
            "dataset": "demo",
            "bbox": [1, 51, 4, 52]
        },
        "output": {
            "datasetId": "demo_subset",
            "title": "My demo subset"
        }
    }
}

JOB_REQUEST_BODY_SCHEMA = {
    "description": "Compute job request",
    "required": True,
    "content": {
        "application/json": {
            "schema": JOB_REQUEST_SCHEMA
        }
    }
}

PATH_PARAM_OP_ID = {
    "name": "operationId",
    "in": "path",
    "description": "Operation identifier",
    "schema": {"type": "string"}
}

PATH_PARAM_JOB_ID = {
    "name": "jobId",
    "in": "path",
    "description": "Job identifier",
    "schema": {"type": "integer"}
}


@api.route('/compute/operations')
class ComputeOperationsHandler(ApiHandler[ComputeContext]):

    @api.operation(
        operation_id="getComputeOperations",
        summary="Get available compute operations.",
        responses=OP_LIST_RESPONSES_SCHEMA
    )
    def get(self):
        self.response.finish(get_compute_operations(self.ctx))


# noinspection PyPep8Naming
@api.route('/compute/operations/{operationId}')
class ComputeOperationHandler(ApiHandler[ComputeContext]):

    @api.operation(
        operation_id="getComputeOperationInfo",
        summary="Get the details of a given compute operation.",
        parameters=[PATH_PARAM_OP_ID],
        responses=OP_RESPONSES_SCHEMA
    )
    def get(self, operationId):
        self.response.finish(get_compute_operation(self.ctx, operationId))


# noinspection PyPep8Naming
@api.route('/compute/jobs')
class ComputeJobsHandler(ApiHandler[ComputeContext]):

    @api.operation(
        operation_id="getComputeJobs",
        summary="Get recent compute jobs.",
        responses=JOB_LIST_RESPONSES_SCHEMA,
    )
    def get(self):
        self.response.finish({
            "jobs": [
                {
                    "jobId": job["jobId"],
                    "operationId": job["request"]["operationId"],
                    "status": job["state"]["status"],
                }
                for job in self.ctx.jobs.values()
            ]
        })

    @api.operation(
        operation_id="scheduleComputeJob",
        summary="Schedule a new compute job.",
        request_body=JOB_REQUEST_BODY_SCHEMA,
        responses=JOB_RESPONSES_SCHEMA,
    )
    def put(self):
        job_request = self.request.json
        # TODO: validate job_request using openapi
        basic_schema = self.put.__openapi__['requestBody']['content'][
            'application/json']['schema']
        try:
            jsonschema.validate(job_request, basic_schema)
            self.response.finish(self.ctx.schedule_job(job_request))
        except jsonschema.ValidationError as e:
            raise ApiError.BadRequest(message=f'{e.message} at {e.json_path}')

# noinspection PyPep8Naming
@api.route('/compute/jobs/{jobId}')
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
