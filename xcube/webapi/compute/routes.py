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

from xcube.server.api import ApiHandler, ApiError
from .api import api
from .context import ComputeContext
from .controllers import get_compute_operations
from .controllers import get_compute_operation


@api.route('/compute/operations')
class ComputeOperationsHandler(ApiHandler[ComputeContext]):

    @api.operation(
        operation_id="getComputeOperations",
        summary="Get available compute operations.",
        parameters=[]
    )
    def get(self):
        self.response.finish(get_compute_operations(self.ctx))


# noinspection PyPep8Naming
@api.route('/compute/operations/{operationId}')
class ComputeOperationHandler(ApiHandler[ComputeContext]):
    """List the available operations."""

    @api.operation(
        operation_id="getComputeOperationInfo",
        summary="Get the details of a given compute operation.",
        parameters=[]
    )
    def get(self, operationId):
        self.response.finish(get_compute_operation(self.ctx, operationId))


# noinspection PyPep8Naming
@api.route('/compute/jobs')
class ComputeJobsHandler(ApiHandler[ComputeContext]):

    @api.operation(
        operation_id="getComputeJobs",
        summary="Get recent compute jobs.",
    )
    def get(self):
        self.response.finish({
            "jobs": list(self.ctx.jobs.values())
        })

    @api.operation(
        operation_id="addComputeJob",
        summary="Start a new compute job.",
    )
    def put(self):
        job_request = self.request.json
        # TODO: validate job_request
        self.response.finish(self.ctx.schedule_job(job_request))


# noinspection PyPep8Naming
@api.route('/compute/jobs/{jobId}')
class ComputeJobHandler(ApiHandler[ComputeContext]):

    @api.operation(
        operation_id="getComputeJob",
        summary="Get details about a compute job.",
    )
    def get(self, jobId: str):
        job = self.ctx.jobs.get(int(jobId))
        if job is None:
            raise ApiError.NotFound(f"job #{jobId} cannot be found")
        self.response.finish(job)

    @api.operation(
        operation_id="cancelComputeJob",
        summary="Cancel an existing compute job.",
    )
    def delete(self, jobId: str):
        self.response.finish(self.ctx.cancel_job(int(jobId)))
