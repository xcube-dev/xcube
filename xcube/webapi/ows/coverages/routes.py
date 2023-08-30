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
import re
from typing import Collection, Optional
import fnmatch
from xcube.server.api import ApiHandler, ApiRequest
from .api import api
from .context import CoveragesContext
from .controllers import get_coverage_as_json, get_coverage_domainset, \
    get_coverage_rangetype, get_collection_metadata, get_coverage_data


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route('/catalog/collections/{collectionId}/coverage')
class CoveragesCoverageHandler(ApiHandler[CoveragesContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id='coveragesCoverage',
                   summary='A coverage in OGC API - Coverages')
    async def get(self, collectionId: str):
        ds_ctx = self.ctx.datasets_ctx
        content_type = get_content_type(
            self.request,
            ['image/tiff', 'application/x-geotiff', 'text/html',
             'application/json', 'application/netcdf', 'application/x-netcdf']
        )
        if content_type == 'text/html':
            result = (f'<html><title>Collection</title><body>'
                      f'<p>{collectionId}</p>'
                      f'</body></html>')
        elif content_type == 'application/json':
            result = get_coverage_as_json(ds_ctx, collectionId)
        else:
            result = get_coverage_data(ds_ctx, collectionId, self.request, content_type)
        return await self.response.finish(result,
                                          content_type=content_type)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route('/catalog/collections/{collectionId}/coverage/domainset')
class CoveragesDomainsetHandler(ApiHandler[CoveragesContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id='coveragesDomainSet',
                   summary='OGC API - Coverages - domain set')
    async def get(self, collectionId: str):
        domain_set = get_coverage_domainset(self.ctx.datasets_ctx, collectionId)
        return self.response.finish(domain_set)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route('/catalog/collections/{collectionId}/coverage/rangetype')
class CoveragesRangetypeHandler(ApiHandler[CoveragesContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id='coveragesDomainSet',
                   summary='OGC API - Coverages - range type')
    async def get(self, collectionId: str):
        range_type = get_coverage_rangetype(self.ctx.datasets_ctx, collectionId)
        return self.response.finish(range_type)


@api.route('/catalog/collections/{collectionId}/coverage/metadata')
class CoveragesMetadataHandler(ApiHandler[CoveragesContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id='coveragesMetadata',
                   summary='OGC API - Coverages - metadata')
    async def get(self, collectionId: str):
        return self.response.finish(get_collection_metadata(
            self.ctx.datasets_ctx, collectionId
        ))


@api.route('/catalog/collections/{collectionId}/coverage/rangeset')
class CoveragesRangesetHandler(ApiHandler[CoveragesContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id='coveragesRangeset',
                   summary='OGC API - Coverages - rangeset')
    async def get(self, collectionId: str):
        self.response.set_status(501)
        return self.response.finish(
            'The rangeset endpoint has been deprecated and is not supported.'
        )


def get_content_type(
        request: ApiRequest, available: Collection[str]
    ) -> Optional[str]:
    if 'f' in request.query:  # overrides headers
        content_type = request.query['f'][0]
        return content_type if available is None or content_type in available else None

    accept = re.split(', *', request.headers.get('Accept'))

    def parse_part(part: str) -> tuple[float, str]:
        if ';q=' in part:
            subparts = part.split(';q=')
            return float(subparts[1]), subparts[0]
        else:
            return 1, part

    type_specs = sorted([parse_part(part) for part in accept], reverse=True)
    types = [ts[1] for ts in type_specs]
    for allowed_type in types:
        for available_type in available:
            # We (ab)use fnmatch to match * wildcards from accept headers
            if fnmatch.fnmatch(available_type, allowed_type):
                return available_type
    return None
