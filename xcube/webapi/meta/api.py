# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api
from .config import CONFIG_SCHEMA
from .context import MetaContext

api = Api(
    "meta",
    description="Server information and maintenance operations",
    config_schema=CONFIG_SCHEMA,
    required_apis=["auth"],
    create_ctx=MetaContext,
)
