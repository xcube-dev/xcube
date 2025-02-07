# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import pkgutil
import sys
from string import Template

from xcube.server.api import ApiError, ApiHandler

from ...constants import LOG
from .api import api
from .context import MetaContext
from .controllers import get_service_info


@api.route("/")
class ServiceInfoHandler(ApiHandler[MetaContext]):
    @api.operation(
        operation_id="getServiceInfo", summary="Get information about the service"
    )
    def get(self):
        self.response.finish(get_service_info(self.ctx))


@api.route("/openapi.html")
class OpenApiHtmlHandler(ApiHandler):
    @api.operation(operation_id="getOpenApiHtml", summary="Show API documentation")
    def get(self):
        include_all = self.request.get_query_arg("all", default=False)
        html_template = pkgutil.get_data(
            "xcube.webapi.meta", "data/openapi.html"
        ).decode("utf-8")
        self.response.finish(
            Template(html_template).substitute(
                open_api_url=self.request.url_for_path(
                    "openapi.json", query="all=1" if include_all else None, reverse=True
                )
            )
        )


@api.route("/openapi.json")
class OpenApiJsonHandler(ApiHandler):
    @api.operation(
        operation_id="getOpenApiJson",
        summary="Get API documentation as OpenAPI 3.0 JSON document",
    )
    def get(self):
        include_all = self.request.get_query_arg("all", default=False)
        self.response.finish(
            self.ctx.get_open_api_doc(include_all=include_all),
            content_type="application/vnd.oai.openapi+json;version=3.0",
        )


@api.route("/maintenance/fail")
class MaintenanceFailHandler(ApiHandler[MetaContext]):
    @api.operation(
        operation_id="forceError",
        summary="Force a request error (for testing)",
        parameters=[
            {
                "name": "code",
                "in": "query",
                "description": "HTTP status code",
                "schema": {
                    "type": "integer",
                    "minimum": 400,
                },
            },
            {
                "name": "message",
                "in": "query",
                "description": "HTTP error message",
                "schema": {
                    "type": "string",
                },
            },
        ],
    )
    def get(self):
        """If *code* is given, the operation fails with
        that HTTP status code. Otherwise, the operation causes
        an internal server error.
        """
        code = self.request.get_query_arg("code", type=int)
        message = self.request.get_query_arg(
            "message", default="Error! No worries, this is just a test."
        )
        LOG.warning("Forcing error in request...")
        if code is None:
            raise RuntimeError(message)
        else:
            raise ApiError(code, message=message)


@api.route("/maintenance/update")
class MaintenanceUpdateHandler(ApiHandler[MetaContext]):
    @api.operation(
        operation_id="updateServer",
        summary="Force server update,"
        " updates configuration, and"
        " resets all resource caches.",
    )
    async def get(self):
        LOG.warning("Forcing server update...")
        # self.ctx.call_later(0,
        #                     self.ctx.server_ctx.server.update,
        #                     self.ctx.config)
        # self.ctx.server_ctx.server.update(self.ctx.config)
        await self.ctx.run_in_executor(
            None, self.ctx.server_ctx.server.update, self.ctx.config
        )
        return self.response.finish(dict(status="OK"))


@api.route("/maintenance/kill")
class MaintenanceUpdateHandler(ApiHandler[MetaContext]):
    @api.operation(
        operation_id="killServer", summary="Force server to shut down immediately."
    )
    def get(self):
        LOG.warning("Forcing server death...")
        sys.exit(0)


@api.route("/maintenance/oom")
class MaintenanceUpdateHandler(ApiHandler[MetaContext]):
    @api.operation(
        operation_id="forceOutOfMemory",
        summary="Force an out-of-memory error (for testing).",
    )
    def get(self):
        LOG.warning("Forcing out-of-memory error...")
        my_bytes = []
        from itertools import count

        for i in count(start=0, step=1):
            my_bytes.append(bytes(2**i))
