# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api
from .context import TilesContext

api = Api(
    "tiles",
    description="xcube Tiles API",
    required_apis=["datasets"],
    create_ctx=TilesContext,
)
