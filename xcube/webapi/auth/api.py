# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api

from .config import AuthConfig
from .context import AuthContext

api = Api(
    "auth",
    description="xcube Auth API",
    config_schema=AuthConfig.get_schema(),
    create_ctx=AuthContext,
)
