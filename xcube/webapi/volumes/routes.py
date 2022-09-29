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
import math

import numpy as np
import pandas as pd

from xcube.server.api import ApiHandler, ApiError
from .api import api
from .context import VolumesContext


# noinspection PyPep8Naming
@api.route('/volumes/{datasetId}/{varName}')
class VolumesContextHandler(ApiHandler[VolumesContext]):
    @api.operation(operation_id='getVolume',
                   summary="Get the volume data for a variable"
                           " in NRRD format.",
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
                           "name": "bbox",
                           "in": "query",
                           "description": 'Bounding box',
                           "schema": {
                               "type": "string",
                           }
                       },
                       {
                           "name": "startDate",
                           "in": "query",
                           "description": "Start date",
                           "schema": {
                               "type": "string",
                               "format": "datetime"
                           }
                       },
                       {
                           "name": "endDate",
                           "in": "query",
                           "description": "End date",
                           "schema": {
                               "type": "string",
                               "format": "datetime"
                           }
                       },
                   ])
    async def get(self, datasetId: str, varName: str):
        bbox = self.request.get_query_arg('bbox',
                                          type=str,
                                          default='')
        if bbox:
            try:
                x1, y1, x2, y2 = map(float, [p.strip()
                                             for p in bbox.split(',')])
            except (ValueError, TypeError):
                raise ApiError.BadRequest("Invalid bbox")
            bbox = x1, y1, x2, y2
        else:
            bbox = None

        start_date = self.request.get_query_arg('startDate',
                                                type=pd.Timestamp,
                                                default=None)
        end_date = self.request.get_query_arg('endDate',
                                              type=pd.Timestamp,
                                              default=None)

        from xcube.core.select import select_subset

        dataset = self.ctx.datasets_ctx.get_dataset(
            datasetId,
            expected_var_names=[varName]
        )
        var = select_subset(
            dataset,
            var_names=[varName],
            bbox=bbox,
            time_range=[start_date,
                        end_date] if start_date or end_date else None
        )[varName]

        if var.ndim != 3:
            raise ApiError.BadRequest(
                f'Variable must be 3-D, got {var.ndim}-D'
            )

        values = var.astype(dtype=np.float32).values
        values = np.where(np.isnan(values), 0.0, values)
        data = values.tobytes(order='C')

        size_z, size_y, size_x = var.shape

        block_size = 1024 * 1024
        num_blocks = math.ceil(len(data) / block_size)

        nrrd_header = (
            "NRRD0004\n"
            "# NRRD 4 Format, "
            "see http://teem.sourceforge.net/nrrd/format.html\n"
            "type: float\n"
            "dimension: 3\n"
            f"sizes: {size_x} {size_y} {size_z}\n"
            "encoding: raw\n"
            "endian: little\n"
            "space directions: (0.1,0,0) (0,0.1,0) (0,0,10)\n"
            "space origin: (0,0,0)\n"
            "\n"
        )

        self.response.set_header('Content-Type', 'application/octet-stream')
        self.response.set_header('Cache-Control', 'max-age=1')
        self.response.write(bytes(nrrd_header, "utf-8"))
        import time
        for i in range(num_blocks):
            self.response.write(
                data[i * block_size: i * block_size + block_size]
            )
            time.sleep(1 / 10000)

        await self.response.finish()
