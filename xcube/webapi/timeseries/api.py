# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import Api

from .context import TimeSeriesContext

api = Api(
    "timeseries",
    description="xcube Timeseries API",
    required_apis=["datasets"],
    create_ctx=TimeSeriesContext,
)
