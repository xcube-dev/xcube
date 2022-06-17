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


# TODO (forman): rename route path
#   to "/tiles/{datasetId}/{varName}/{z}/{y}/{x}"

# noinspection PyPep8Naming
@api.route('/datasets/{datasetId}/vars/{varName}/tiles2/{z}/{y}/{x}')
class TilesHandler(ApiHandler[TilesContext]):
    @api.operation(operation_id='getTile',
                   summary="Get the image tile for a variable"
                           " and given tile grid coordinates.",
                   parameters=[
                       {
                           "name": "datasetId",
                           "in": "path",
                           "description": "Dataset identifier",
                           "schema": {
                               "type": "string",
                           }
                       },
                       {
                           "name": "varName",
                           "in": "path",
                           "description": "Name of variable in dataset",
                           "schema": {
                               "type": "string",
                           }
                       },
                       {
                           "name": "z",
                           "in": "path",
                           "description": "The tile grid's z-coordinate",
                           "schema": {
                               "type": "integer",
                           }
                       },
                       {
                           "name": "y",
                           "in": "path",
                           "description": "The tile grid's y-coordinate",
                           "schema": {
                               "type": "integer",
                           }
                       },
                       {
                           "name": "x",
                           "in": "path",
                           "description": "The tile grid's x-coordinate",
                           "schema": {
                               "type": "integer",
                           }
                       },
                       {
                           "name": "crs",
                           "in": "query",
                           "description": "The tile grid's spatial CRS",
                           "schema": {
                               "type": "string",
                               "enum": ["EPSG:3857", "CRS84"],
                               "default": "CRS84"
                           }
                       },
                       {
                           "name": "vmin",
                           "in": "query",
                           "description": "Minimum value of variable"
                                          " for color mapping",
                           "schema": {
                               "type": "number",
                               "default": 0
                           }
                       },
                       {
                           "name": "vmax",
                           "in": "query",
                           "description": "Maximum value of variable"
                                          " for color mapping",
                           "schema": {
                               "type": "number",
                               "default": 1
                           }
                       },
                       {
                           "name": "cbar",
                           "in": "query",
                           "description": "Name of the color bar"
                                          " for color mapping",
                           "schema": {
                               "type": "string",
                               "default": "bone"
                           }
                       },
                       {
                           "name": "format",
                           "in": "query",
                           "description": "Image format",
                           "schema": {
                               "type": "string",
                               "enum": ["png", "image/png"],
                               "default": "png"
                           }
                       },
                       {
                           "name": "retina",
                           "in": "query",
                           "description": "Returns tiles of size"
                                          " 512 instead of 256",
                           "schema": {
                               "type": "boolean"
                           }
                       },
                   ])
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
