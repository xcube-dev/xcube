# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api

from .config import CONFIG_SCHEMA
from .context import PlacesContext

api = Api(
    "places",
    description="xcube Places API",
    config_schema=CONFIG_SCHEMA,
    required_apis=["auth"],
    create_ctx=PlacesContext,
)
