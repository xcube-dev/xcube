# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiHandler

from ..datasets import (
    PATH_PARAM_DATASET_ID,
    PATH_PARAM_VAR_NAME,
    QUERY_PARAM_CMAP,
    QUERY_PARAM_CRS,
    QUERY_PARAM_NORM,
    QUERY_PARAM_VMAX,
    QUERY_PARAM_VMIN,
)
from .api import api
from .context import TilesContext
from .controllers import compute_ml_dataset_tile

PATH_PARAM_X = {
    "name": "x",
    "in": "path",
    "description": "The tile grid's x-coordinate",
    "schema": {
        "type": "integer",
    },
}

PATH_PARAM_Y = {
    "name": "y",
    "in": "path",
    "description": "The tile grid's y-coordinate",
    "schema": {
        "type": "integer",
    },
}

PATH_PARAM_Z = {
    "name": "z",
    "in": "path",
    "description": "The tile grid's z-coordinate",
    "schema": {
        "type": "integer",
    },
}

QUERY_PARAM_TIME = {
    "name": "time",
    "in": "query",
    "description": 'Optional time coordinate using format "YYYY-MM-DD hh:mm:ss"',
    "schema": {"type": "string", "format": "datetime"},
}

QUERY_PARAM_FORMAT = {
    "name": "format",
    "in": "query",
    "description": "Image format",
    "schema": {"type": "string", "enum": ["png", "image/png"], "default": "png"},
}

QUERY_PARAM_RETINA = {
    "name": "retina",
    "in": "query",
    "description": "Returns tiles of size 512 instead of 256",
    "schema": {"type": "boolean"},
}

TILE_PARAMETERS = [
    PATH_PARAM_DATASET_ID,
    PATH_PARAM_VAR_NAME,
    PATH_PARAM_Z,
    PATH_PARAM_Y,
    PATH_PARAM_X,
    QUERY_PARAM_CRS,
    QUERY_PARAM_VMIN,
    QUERY_PARAM_VMAX,
    QUERY_PARAM_CMAP,
    QUERY_PARAM_NORM,
    QUERY_PARAM_TIME,
    QUERY_PARAM_FORMAT,
    QUERY_PARAM_RETINA,
]


# noinspection PyPep8Naming
@api.route("/tiles/{datasetId}/{varName}/{z}/{y}/{x}")
class TilesHandler(ApiHandler[TilesContext]):
    @api.operation(
        operation_id="getTile",
        summary="Get the image tile for a variable and given tile grid coordinates.",
        parameters=TILE_PARAMETERS,
    )
    async def get(self, datasetId: str, varName: str, z: str, y: str, x: str):
        tile = await self.ctx.run_in_executor(
            None,
            compute_ml_dataset_tile,
            self.ctx,
            datasetId,
            varName,
            None,
            x,
            y,
            z,
            {k: v[0] for k, v in self.request.query.items()},
        )
        self.response.set_header("Content-Type", "image/png")
        await self.response.finish(tile)
