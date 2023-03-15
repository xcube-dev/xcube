# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from xcube.server.api import ApiHandler
from .api import api
from .context import TilesContext
from .controllers import compute_ml_dataset_tile
from ..datasets import PATH_PARAM_DATASET_ID
from ..datasets import PATH_PARAM_VAR_NAME
from ..datasets import QUERY_PARAM_CBAR
from ..datasets import QUERY_PARAM_CRS
from ..datasets import QUERY_PARAM_VMAX
from ..datasets import QUERY_PARAM_VMIN

PATH_PARAM_X = {
    "name": "x",
    "in": "path",
    "description": "The tile grid's x-coordinate",
    "schema": {
        "type": "integer",
    }
}

PATH_PARAM_Y = {
    "name": "y",
    "in": "path",
    "description": "The tile grid's y-coordinate",
    "schema": {
        "type": "integer",
    }
}

PATH_PARAM_Z = {
    "name": "z",
    "in": "path",
    "description": "The tile grid's z-coordinate",
    "schema": {
        "type": "integer",
    }
}

QUERY_PARAM_TIME = {
    "name": "time",
    "in": "query",
    "description": "Optional time coordinate using format"
                   " \"YYYY-MM-DD hh:mm:ss\"",
    "schema": {
        "type": "string",
        "format": "datetime"
    }
}

QUERY_PARAM_FORMAT = {
    "name": "format",
    "in": "query",
    "description": "Image format",
    "schema": {
        "type": "string",
        "enum": ["png", "image/png"],
        "default": "png"
    }
}

QUERY_PARAM_RETINA = {
    "name": "retina",
    "in": "query",
    "description": "Returns tiles of size"
                   " 512 instead of 256",
    "schema": {
        "type": "boolean"
    }
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
    QUERY_PARAM_CBAR,
    QUERY_PARAM_TIME,
    QUERY_PARAM_FORMAT,
    QUERY_PARAM_RETINA,
]


# noinspection PyPep8Naming
class TilesHandler(ApiHandler[TilesContext]):
    async def get(self,
                  datasetId: str,
                  varName: str,
                  z: str, y: str, x: str):
        tile = await self.ctx.run_in_executor(
            None,
            compute_ml_dataset_tile,
            self.ctx,
            datasetId,
            varName,
            None,
            x, y, z,
            {k: v[0] for k, v in self.request.query.items()}
        )
        self.response.set_header('Content-Type', 'image/png')
        await self.response.finish(tile)


# noinspection PyPep8Naming
@api.route('/tiles/{datasetId}/{varName}/{z}/{y}/{x}')
class NewTilesHandler(TilesHandler):
    @api.operation(operation_id='getTile',
                   summary="Get the image tile for a variable"
                           " and given tile grid coordinates.",
                   parameters=TILE_PARAMETERS)
    async def get(self,
                  datasetId: str,
                  varName: str,
                  z: str, y: str, x: str):
        await super().get(datasetId, varName, z, y, x)


# For backward compatibility only
# noinspection PyPep8Naming
@api.route('/datasets/{datasetId}/vars/{varName}/tiles2/{z}/{y}/{x}')
class OldTilesHandler(TilesHandler):
    """Deprecated. Use /tiles/{datasetId}/{varName}/{z}/{y}/{x}."""
    @api.operation(operation_id='getDatasetVariableTile',
                   tags=['datasets'],
                   summary="Get the image tile for a variable"
                           " and given tile grid coordinates. (deprecated)",
                   parameters=TILE_PARAMETERS)
    async def get(self,
                  datasetId: str,
                  varName: str,
                  z: str, y: str, x: str):
        await super().get(datasetId, varName, z, y, x)
