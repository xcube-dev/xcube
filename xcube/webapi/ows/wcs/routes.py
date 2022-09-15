# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
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
from .context import WcsContext
from .controllers import get_wcs_capabilities_xml, get_describe_coverage_xml, \
    get_coverage, translate_to_generator_request, CoverageRequest

WCS_VERSION = '1.0.0'


@api.route('/wcs/1.0.0/WCSCapabilities.xml')
class WcsCapabilitiesXmlHandler(ApiHandler[WcsContext]):
    @api.operation(operationId='getWcsCapabilities',
                   summary='Gets the WCS capabilities as XML document')
    async def get(self):
        self.request.make_query_lower_case()
        capabilities = await self.ctx.run_in_executor(
            None,
            get_wcs_capabilities_xml,
            self.ctx,
            self.request.base_url
        )
        self.response.set_header('Content-Type', 'application/xml')
        await self.response.finish(capabilities)


@api.route('/wcs/kvp')
class WcsKvpHandler(ApiHandler[WcsContext]):
    @api.operation(operationId='invokeWcsMethodFromKvp',
                   summary='Invokes the WCS by key-value pairs')
    async def get(self):
        self.request.make_query_lower_case()
        service = self.request.get_query_arg('service', default='WCS')
        if service != 'WCS':
            raise ApiError.BadRequest(
                'value for "service" parameter must be "WCS"'
            )
        request = self.request.get_query_arg('request')
        if request is None:
            request = self.request.get_query_arg('REQUEST')
        if request == "GetCapabilities":
            wcs_version = self.request.get_query_arg(
                "version", default=WCS_VERSION
            )
            if wcs_version != WCS_VERSION:
                raise ApiError.BadRequest(
                    f'value for "version" parameter must be "{WCS_VERSION}"'
                )
            capabilities_xml = await self.ctx.run_in_executor(
                None,
                get_wcs_capabilities_xml,
                self.ctx,
                self.request.base_url
            )
            self.response.set_header('Content-Type', 'application/xml')
            await self.response.finish(capabilities_xml)

        elif request == 'DescribeCoverage':
            wcs_version = self.request.get_query_arg('version',
                                                     default=WCS_VERSION)
            if wcs_version != WCS_VERSION:
                raise ApiError.BadRequest(
                    f'value for "version" parameter must be "{WCS_VERSION}"'
                )
            coverages = self.request.get_query_arg("coverage")
            if not coverages:
                coverages = self.request.get_query_arg("COVERAGE")
            if coverages:
                coverages = coverages.split(',')

            describe_coverage_xml = await self.ctx.run_in_executor(
                None,
                get_describe_coverage_xml,
                self.ctx,
                coverages
            )
            self.response.set_header('Content-Type', 'application/xml')
            await self.response.finish(describe_coverage_xml)
        elif request == "GetCoverage":
            wcs_version = self.request.get_query_arg('version',
                                                     default=WCS_VERSION)
            if wcs_version != WCS_VERSION:
                raise ApiError.BadRequest(
                    f'value for "version" parameter must be "{WCS_VERSION}"'
                )
            coverage = self.request.get_query_arg('coverage')
            if not coverage:
                raise ApiError.BadRequest(
                    f'missing query argument "coverage"'
                )
            request_crs = self.request.get_query_arg('crs')
            if not request_crs:
                raise ApiError.BadRequest(
                    f'missing query argument "crs"'
                )
            response_crs = self.request.get_query_arg('response_crs',
                                                      default=request_crs)
            time = self.request.get_query_arg('time')

            cov_req = CoverageRequest({
                'COVERAGE': coverage,
                'CRS': response_crs,
                'TIME': time
            })
            gen_request = translate_to_generator_request(self.ctx, cov_req)
            coverage = await self.ctx.run_in_executor(
                None,
                get_coverage,
                self.ctx,
                gen_request
            )
            # self.response.set_header('Content-Type', 'application/xml')
            await self.response.finish(get_coverage)
        else:
            raise ApiError.BadRequest(
                f'invalid request type "{request}"'
            )


def _query_to_dict(request):
    return {k: v[0] for k, v in request.query.items()}
