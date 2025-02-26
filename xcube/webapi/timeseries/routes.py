# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import pandas as pd

from xcube.server.api import ApiHandler

from ..datasets import PATH_PARAM_DATASET_ID, PATH_PARAM_VAR_NAME
from .api import api
from .context import TimeSeriesContext
from .controllers import get_time_series


# noinspection PyPep8Naming
@api.route("/timeseries/{datasetId}/{varName}")
class TimeseriesHandler(ApiHandler[TimeSeriesContext]):
    @api.operation(
        operation_id="getTimeSeries",
        summary="Get the time-series for a variable and given GeoJSON object.",
        parameters=[
            PATH_PARAM_DATASET_ID,
            PATH_PARAM_VAR_NAME,
            {
                "name": "aggMethods",
                "in": "query",
                "description": "Comma-separated list of"
                " aggregation methods."
                " Valid methods are"
                ' "mean", "median", "std",'
                ' "min", "max", "count"',
                "schema": {
                    "type": "string",
                },
            },
            {
                "name": "startDate",
                "in": "query",
                "description": "Start timestamp",
                "schema": {"type": "string", "format": "datetime"},
            },
            {
                "name": "endDate",
                "in": "query",
                "description": "End timestamp",
                "schema": {"type": "string", "format": "datetime"},
            },
            {
                "name": "tolerance",
                "in": "query",
                "description": "Time tolerance in seconds that"
                " expands the given time range",
                "schema": {
                    "type": "string",
                    "default": 1.0,
                },
            },
            {
                "name": "maxValids",
                "in": "query",
                "description": "Maximum number of valid"
                " time-series values"
                " to be returned.",
                "schema": {
                    "type": "integer",
                },
            },
        ],
    )
    async def post(self, datasetId: str, varName: str):
        geo_json_object = self.request.json
        agg_methods = self.request.get_query_arg("aggMethods", type=str, default=None)
        agg_methods = agg_methods.split(",") if agg_methods else None
        start_date = self.request.get_query_arg(
            "startDate", type=pd.Timestamp, default=None
        )
        end_date = self.request.get_query_arg(
            "endDate", type=pd.Timestamp, default=None
        )
        tolerance = self.request.get_query_arg("tolerance", type=float, default=1.0)
        max_valids = self.request.get_query_arg("maxValids", type=int, default=None)
        result = await self.ctx.run_in_executor(
            None,
            get_time_series,
            self.ctx,
            datasetId,
            varName,
            geo_json_object,
            agg_methods,
            start_date,
            end_date,
            tolerance,
            max_valids,
        )
        self.response.set_header("Content-Type", "application/json")
        await self.response.finish(dict(result=result))
