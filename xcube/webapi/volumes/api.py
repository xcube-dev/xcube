# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api
from .config import CONFIG_SCHEMA
from .context import VolumesContext

api = Api(
    "volumes",
    description="xcube Volumes API",
    required_apis=["datasets"],
    config_schema=CONFIG_SCHEMA,
    create_ctx=VolumesContext,
)
