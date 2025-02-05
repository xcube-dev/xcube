# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
import fnmatch
import json
import re
from collections.abc import Collection
from typing import Optional

from xcube.server.api import ApiError, ApiHandler, ApiRequest

from .api import api
from .context import CoveragesContext
from .controllers import (
    get_collection_metadata,
    get_coverage_as_json,
    get_coverage_data,
    get_coverage_domainset,
    get_coverage_rangetype,
)

PATH_PREFIX = "/ogc"
_COVERAGE_PREFIX = PATH_PREFIX + "/collections/{collectionId}/coverage"


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(_COVERAGE_PREFIX, slash=True)
class CoveragesCoverageHandler(ApiHandler[CoveragesContext]):
    """
    Return coverage data

    This is the main coverage endpoint: the one that returns the actual
    data (as opposed to metadata), as TIFF or NetCDF. It can also provide
    representations in HTML and JSON.
    """

    # noinspection PyPep8Naming
    @api.operation(
        operation_id="coveragesCoverage",
        summary="A coverage in OGC API - Coverages",
    )
    async def get(self, collectionId: str):
        ds_ctx = self.ctx.datasets_ctx

        # The single-component type specifiers aren't RFC2045-compliant,
        #  but the OGC API - Coverages draft allows them in the f parameter.
        available_types = [
            "png",
            "image/png",
            "image/tiff",
            "application/x-geotiff",
            "geotiff",
            "application/netcdf",
            "application/x-netcdf",
            "netcdf",
            "html",
            "text/html",
            "json",
            "application/json",
            # TODO: support covjson
        ]
        content_type = negotiate_content_type(self.request, available_types)
        content_bbox = content_crs = None
        if content_type is None:
            raise ApiError.UnsupportedMediaType(
                f"Available media types: {', '.join(available_types)}\n"
            )
        elif content_type in {"text/html", "html"}:
            result = (
                "<html><title>Collection</title><body><pre>\n"
                + json.dumps(get_coverage_as_json(ds_ctx, collectionId), indent=2)
                + "\n</pre></body></html>"
            )
        elif content_type in {"application/json", "json"}:
            result = get_coverage_as_json(ds_ctx, collectionId)
        else:
            result, content_bbox, content_crs = get_coverage_data(
                ds_ctx, collectionId, self.request.query, content_type
            )

        if content_bbox is not None:
            self.response.set_header("Content-Bbox", ",".join(map(str, content_bbox)))
        if content_crs is not None:
            self.response.set_header("Content-Crs", f"[{content_crs.to_string()}]")
        return await self.response.finish(result, content_type=content_type)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(_COVERAGE_PREFIX + "/domainset", slash=True)
class CoveragesDomainsetHandler(ApiHandler[CoveragesContext]):
    """Describe the domain set of a coverage

    The domain set is the set of input parameters (e.g. geographical extent,
    time span) for which the coverage is defined.
    """

    # noinspection PyPep8Naming
    @api.operation(
        operation_id="coveragesDomainSet",
        summary="OGC API - Coverages - domain set",
    )
    async def get(self, collectionId: str):
        domain_set = get_coverage_domainset(self.ctx.datasets_ctx, collectionId)
        return self.response.finish(domain_set)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(_COVERAGE_PREFIX + "/rangetype", slash=True)
class CoveragesRangetypeHandler(ApiHandler[CoveragesContext]):
    """Describe the range type of a coverage

    The range type describes the types of the data contained in the range
    of this coverage. For a data cube, these would typically correspond to
    the types of the variables or bands.
    """

    # noinspection PyPep8Naming
    @api.operation(
        operation_id="coveragesRangeType",
        summary="OGC API - Coverages - range type",
    )
    async def get(self, collectionId: str):
        range_type = get_coverage_rangetype(self.ctx.datasets_ctx, collectionId)
        return self.response.finish(range_type)


@api.route(_COVERAGE_PREFIX + "/metadata", slash=True)
class CoveragesMetadataHandler(ApiHandler[CoveragesContext]):
    """Return coverage metadata

    The metadata is taken from the source dataset's attributes
    """

    # noinspection PyPep8Naming
    @api.operation(
        operation_id="coveragesMetadata",
        summary="OGC API - Coverages - metadata",
    )
    async def get(self, collectionId: str):
        return self.response.finish(
            get_collection_metadata(self.ctx.datasets_ctx, collectionId)
        )


@api.route(_COVERAGE_PREFIX + "/rangeset", slash=True)
class CoveragesRangesetHandler(ApiHandler[CoveragesContext]):
    """Handle rangeset endpoint with a "not allowed" response

    This endpoint has been deprecated
    (see https://github.com/m-mohr/geodatacube-api/pull/1 ),
    but we handle it to make clear that its non-implementation is a deliberate
    decision.
    """

    # noinspection PyPep8Naming
    @api.operation(
        operation_id="coveragesRangeset",
        summary="OGC API - Coverages - rangeset",
    )
    async def get(self, collectionId: str):
        self.response.set_status(405)
        self.response.set_header("Allow", "")  # no methods supported
        return self.response.finish(
            "The rangeset endpoint has been deprecated and is not supported."
        )


def negotiate_content_type(
    request: ApiRequest, available: Collection[str]
) -> Optional[str]:
    """Determine a MIME content type based on client and server capabilities

    Client preferences are determined both from the standard HTTP Accept
    header provided by the client, and by an optional `f` parameter which
    can override the Accept header.

    Args:
        request: HTTP request with Accept header and/or f parameter
        available: List of MIME types that the server can produce

    Returns:
        the MIME type that is most acceptable to both client and server,
        or None if there is no MIME type acceptable to both
    """
    if "f" in request.query:  # overrides headers
        content_type = request.query["f"][0]
        return content_type if available is None or content_type in available else None

    accept = re.split(", *", request.headers.get("Accept"))

    def parse_part(part: str) -> tuple[float, str]:
        if ";q=" in part:
            subparts = part.split(";q=")
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
