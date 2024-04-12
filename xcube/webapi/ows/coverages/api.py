# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api
from xcube.webapi.ows.coverages.context import CoveragesContext

api = Api(
    "ows.coverages",
    description="OGC API - Coverages",
    required_apis=["datasets"],
    create_ctx=CoveragesContext,
    config_schema=None,
)
