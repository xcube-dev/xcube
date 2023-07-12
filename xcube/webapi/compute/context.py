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

import importlib
import concurrent.futures
import datetime
from typing import Dict, Any

import xarray as xr

from xcube.server.api import Context
from xcube.webapi.common.context import ResourcesContext
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.places import PlacesContext
from xcube.constants import LOG
from .op.info import OpInfo
from .op.registry import OpRegistry
from .op.registry import OP_REGISTRY

# Register default operations:
importlib.import_module("xcube.webapi.compute.operations")


LocalExecutor = concurrent.futures.ThreadPoolExecutor


class ComputeContext(ResourcesContext):

    def __init__(self,
                 server_ctx: Context,
                 op_registry: OpRegistry = OP_REGISTRY):
        super().__init__(server_ctx)
        self._datasets_ctx: DatasetsContext \
            = server_ctx.get_api_ctx("datasets")
        assert isinstance(self._datasets_ctx, DatasetsContext)
        self._places_ctx: PlacesContext \
            = server_ctx.get_api_ctx("places")
        assert isinstance(self._places_ctx, PlacesContext)

        self._op_registry = op_registry
        assert isinstance(self._op_registry, OpRegistry)

        compute_config = server_ctx.config.get("Compute", {})
        max_workers = compute_config.get("MaxWorkers", 3)

        self.next_job_id = 0
        self.jobs: Dict[int, Any] = {}
        self.job_futures: Dict[int, concurrent.futures.Future] = {}
        self.job_executor = LocalExecutor(max_workers=max_workers,
                                          thread_name_prefix='xcube-job-')

    def on_dispose(self):
        self.job_executor.shutdown(cancel_futures=True)

    @property
    def op_registry(self) -> OpRegistry:
        return self._op_registry

    @property
    def datasets_ctx(self) -> DatasetsContext:
        return self._datasets_ctx

    @property
    def places_ctx(self) -> PlacesContext:
        return self._places_ctx

    def schedule_job(self, job_request: Dict[str, Any]):
        """Schedule a new job given by *job_request*,
        which is expected to be validated already.

        Job status transitions:
            scheduled
            scheduled --> cancelled
            scheduled --> running
            scheduled --> running --> completed
            scheduled --> running --> failed
            scheduled --> running --> cancelled
        """

        with self.rlock:
            job_id = self.next_job_id
            self.next_job_id += 1

        job = {
            "id": job_id,
            "request": job_request,
            "state": {
                "status": "scheduled"
            },
            "createTime": datetime.datetime.utcnow().isoformat(),
            "startTime": None,
        }

        LOG.info(f"Scheduled job #{job_id}")
        self.jobs[job_id] = job
        job_future = self.job_executor.submit(self.invoke_job, job_id)
        self.job_futures[job_id] = job_future

        return job

    def invoke_job(self, job_id: int):
        job = self.jobs.get(job_id)
        if job is None:
            return

        if job["state"]["status"] == "cancelled":
            return

        job_request = job["request"]

        op_id = job_request["operationId"]
        output_ds_id = job_request["datasetId"]
        parameters = job_request.get("parameters", {})

        LOG.info(f"Started job #{job_id}")
        job["state"]["status"] = "running"

        op = self.op_registry.get_op(op_id)
        op_info = OpInfo.get_op_info(op)
        param_py_types = op_info.effective_param_py_types

        parameters = parameters.copy()
        for param_name, param_py_type in param_py_types.items():
            if param_py_type is xr.Dataset:
                input_ds_id = parameters.get(param_name)
                if input_ds_id is not None:
                    input_ds = self._datasets_ctx.get_dataset(input_ds_id)
                    parameters[param_name] = input_ds

        try:
            output_ds = op(**parameters)
        except Exception as e:
            LOG.error(f"Job #{job_id} failed:", e)
            job["state"]["status"] = "failed"
            job["state"]["error"] = {
                "message": str(e),
                "type": type(e).__name__
            }
            return

        if job["state"]["status"] == "cancelled":
            return

        # TODO: get other dataset properties from "output"
        self.datasets_ctx.add_dataset(
            output_ds,
            ds_id=output_ds_id
        )

        LOG.info(f"Completed job #{job_id}")
        job["state"]["status"] = "completed"

    def cancel_job(self, job_id: int):
        job = self.jobs.get(job_id)
        if job is None:
            return

        LOG.info(f"Job #{job_id} cancelled:")
        job["state"]["status"] = "cancelled"

        future = self.job_futures.pop(job_id, None)
        if future is not None:
            future.cancel()
