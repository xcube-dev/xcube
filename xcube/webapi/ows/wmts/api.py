# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api
from .context import WmtsContext

api = Api(
    "ows.wmts",
    description="xcube OGC WMTS API",
    required_apis=["tiles", "datasets"],
    create_ctx=WmtsContext,
)
