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
from typing import Collection, Optional

from xcube.server.api import ApiHandler, ApiRequest
import re

from .api import api
from .context import CoveragesContext
from .controllers import get_coverage_as_json, get_coverage_domainset, \
    get_coverage_rangetype, get_collection_metadata, get_collection_envelope, \
    get_dataset, get_coverage_as_tiff


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route('/catalog/collections/{collectionId}/coverage')
class CoveragesCoverageHandler(ApiHandler[CoveragesContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id='coveragesCoverage',
                   summary='A coverage in OGC API - Coverages')
    async def get(self, collectionId: str):
        ds_ctx = self.ctx.datasets_ctx
        content_type = get_content_type(self.request)
        actual_content_type = get_content_type(self.request)
        if content_type == 'text/html':
            result = (f'<html><title>Collection</title><body>'
                      f'<p>{collectionId}</p>'
                      f'</body></html>')
        elif content_type == 'application/json':
            result = get_coverage_as_json(ds_ctx, collectionId)
        else:
            actual_content_type='image/tiff'
            result = get_coverage_as_tiff()
        return await self.response.finish(result,
                                          content_type=actual_content_type)


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


def get_content_type(
        request: ApiRequest, available: Optional[Collection[str]] = None
    ) -> Optional[str]:
    if 'f' in request.query:  # overrides headers
        content_type = request.query['f'][0]
        return content_type if content_type in available else None

    accept = re.split(', *', request.headers.get('Accept'))

    def parse_part(part: str) -> tuple[float, str]:
        if ';q=' in part:
            subparts = part.split(';q=')
            return float(subparts[1]), subparts[0]
        else:
            return 1, part
    type_specs = sorted([parse_part(part) for part in accept], reverse=True)
    types = [ts[1] for ts in type_specs]
    if available is not None:
        types = list(filter(lambda t: t in available, types))
    return types[0] if len(types) > 0 else None

