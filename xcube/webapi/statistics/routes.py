# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiHandler
from xcube.util.undefined import UNDEFINED

from ..datasets.routes import PATH_PARAM_DATASET_ID, PATH_PARAM_VAR_NAME

from .api import api
from .context import StatisticsContext
from .controllers import compute_statistics
import logging
from collections.abc import Hashable
from typing import Any
from xcube.core.tile import get_non_spatial_labels

_logger = logging.getLogger(__name__)

QUERY_PARAM_X = {
    "name": "lon",
    "in": "query",
    "description": "Longitude in decimal degree",
    "required": True,
    "schema": {"type": "number", "minimum": -180, "maximum": 180},
}

QUERY_PARAM_Y = {
    "name": "lat",
    "in": "query",
    "description": "Latitude in decimal degree",
    "required": True,
    "schema": {"type": "number", "minimum": -90, "maximum": 90},
}

QUERY_PARAM_TIME = {
    "name": "time",
    "in": "query",
    "description": 'Timestamp using format "YYYY-MM-DD hh:mm:ss"',
    "required": False,
    "schema": {"type": "string", "format": "datetime"},
}


# noinspection PyPep8Naming
@api.route("/statistics/{datasetId}/{varName}")
class StatisticsHandler(ApiHandler[StatisticsContext]):
    @api.operation(
        operation_id="getValue",
        summary=(
            "Get the value of a dataset variable for given time stamp and location."
        ),
        parameters=[
            PATH_PARAM_DATASET_ID,
            PATH_PARAM_VAR_NAME,
            QUERY_PARAM_X,
            QUERY_PARAM_Y,
            QUERY_PARAM_TIME,
        ],
    )
    async def get(self, datasetId: str, varName: str):
        lon = self.request.get_query_arg("lon", type=float, default=UNDEFINED)
        lat = self.request.get_query_arg("lat", type=float, default=UNDEFINED)
        non_spatial_dimensions = get_non_spatial_dimensions(
            self.ctx, self.request, datasetId, varName
        )

        trace_perf = self.request.get_query_arg(
            "debug", default=self.ctx.datasets_ctx.trace_perf
        )
        result = await self.ctx.run_in_executor(
            None,
            compute_statistics,
            self.ctx,
            datasetId,
            varName,
            (lon, lat),
            non_spatial_dimensions,
            trace_perf,
        )
        await self.response.finish({"result": result})

    @api.operation(
        operation_id="getStatistics",
        summary=(
            "Get statistics of a dataset variable for given time stamp and geometry."
        ),
        description=(
            "The geometry is passed in the request body in"
            " form of a valid GeoJSON geometry object."
            " The operation returns the count, minimum, maximum, mean,"
            " and standard deviation of a data variable. If a 2-D geometry"
            " is passed, a histogram is returned as well."
        ),
        parameters=[
            PATH_PARAM_DATASET_ID,
            PATH_PARAM_VAR_NAME,
            QUERY_PARAM_TIME,
        ],
    )
    async def post(self, datasetId: str, varName: str):
        non_spatial_dimensions = get_non_spatial_dimensions(
            self.ctx, self.request, datasetId, varName
        )
        trace_perf = self.request.get_query_arg(
            "debug", default=self.ctx.datasets_ctx.trace_perf
        )
        result = await self.ctx.run_in_executor(
            None,
            compute_statistics,
            self.ctx,
            datasetId,
            varName,
            self.request.json,
            non_spatial_dimensions,
            trace_perf,
        )
        await self.response.finish({"result": result})


def get_non_spatial_dimensions(ctx, request, ds_id, var) -> dict[Hashable, Any]:

    try:
        ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_id)
        ds = ml_dataset.get_dataset(0)
        variable = ds[var]
    except KeyError as e:
        _logger.error(f"Failed to retrieve dataset '{ds_id}' or variable '{var}': {e}")
        raise

    variable_dims = variable.dims
    dimensions = {}
    for dim in variable_dims:
        value = request.get_query_arg(str(dim), type=str, default=None)
        if value is not None:
            dimensions[str(dim)] = value

    labels = get_non_spatial_labels(ds, variable, labels=dimensions, logger=_logger)
    return labels
