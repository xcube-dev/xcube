# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api

from .config import CONFIG_SCHEMA
from .context import StacContext

api = Api(
    "ows.stac",
    description="xcube OGC STAC API",
    required_apis=["auth", "datasets"],
    create_ctx=StacContext,
    config_schema=CONFIG_SCHEMA,
)
