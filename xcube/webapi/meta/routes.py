#  The MIT License (MIT)
#  Copyright (c) 2022 by the xcube development team and contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

import os
import pkgutil
from string import Template

from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from xcube.util.versions import get_xcube_versions
from xcube.version import version
from .api import api
from .context import MetaContext


@api.route("/")
class ServiceInfoHandler(ApiHandler[MetaContext]):
    @api.operation(operation_id='getServiceInfo',
                   summary='Get information about the service')
    def get(self):
        api_infos = []
        for other_api in self.ctx.apis:
            api_info = {
                "name": other_api.name,
                "version": other_api.version,
                "description": other_api.description
            }
            api_infos.append({k: v
                              for k, v in api_info.items()
                              if v is not None})

        # TODO (forman): once APIs are configurable, we should
        #   get the server name, description from configuration too
        self.response.finish(dict(
            name="xcube Server",
            description="xcube Server",
            version=version,
            versions=get_xcube_versions(),
            apis=api_infos,
            startTime=self.ctx.start_time,
            currentTime=self.ctx.current_time,
            updateTime=self.ctx.update_time,
            serverProcessId=os.getpid()
        ))


@api.route("/openapi.html")
class OpenApiHtmlHandler(ApiHandler):
    @api.operation(
        operation_id='getOpenApiHtml',
        summary='Show API documentation'
    )
    def get(self):
        html_template = pkgutil.get_data('xcube.webapi.meta.res',
                                         'openapi.html').decode('utf-8')
        self.response.finish(Template(html_template).substitute(
            open_api_url=self.request.url_for_path('openapi.json')
        ))


@api.route("/openapi.json")
class OpenApiJsonHandler(ApiHandler):
    @api.operation(
        operation_id='getOpenApiJson',
        summary='Get API documentation as OpenAPI 3.0 JSON document'
    )
    def get(self):
        self.response.finish(self.ctx.open_api_doc)


@api.route("/error")
class ErrorHandler(ApiHandler):
    @api.operation(operation_id='forceError',
                   summary='Force a server error (for testing)',
                   parameters=[
                       {
                           "name": "code",
                           "in": "query",
                           "description": "HTTP status code",
                           "schema": {
                               "type": "integer",
                               "minimum": 400,
                           }
                       },
                       {
                           "name": "message",
                           "in": "query",
                           "description": "HTTP error message",
                           "schema": {
                               "type": "string",
                           }
                       },
                   ])
    def get(self):
        """If *code* is given, the operation fails with
        that HTTP status code. Otherwise, the operation causes
        an internal server error.
        """
        code = self.request.get_query_arg('code', type=int)
        message = self.request.get_query_arg(
            'message',
            default='Error! No worries, this is just a test.'
        )
        if code is None:
            raise RuntimeError(message)
        else:
            raise ApiError(code, message=message)
