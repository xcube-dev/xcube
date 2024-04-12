# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api
from .config import CONFIG_SCHEMA
from .context import ViewerContext

api = Api(
    "viewer",
    description="xcube Viewer web application",
    create_ctx=ViewerContext,
    config_schema=CONFIG_SCHEMA,
)
