# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api
from .context import S3Context

api = Api(
    "s3",
    description="xcube S3 API emulation",
    required_apis=["datasets"],
    create_ctx=S3Context,
)
