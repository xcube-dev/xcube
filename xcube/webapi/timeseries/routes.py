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

import pandas as pd

from xcube.server.api import ApiHandler
from .api import api
from .context import TimeSeriesContext
from .controllers import get_time_series


# noinspection PyPep8Naming
@api.route('/timeseries/{datasetId}/{varName}')
class TimeseriesHandler(ApiHandler[TimeSeriesContext]):
    @api.operation(operation_id='getTimeSeries',
                   summary="Get the time-series for a variable"
                           " and given GeoJSON object.",
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
                           "name": "aggMethods",
                           "in": "query",
                           "description": 'Comma-separated list of'
                                          ' aggregation methods.'
                                          ' Valid methods are'
                                          ' "mean", "median", "std",'
                                          ' "min", "max", "count"',
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
                       {
                           "name": "maxValids",
                           "in": "query",
                           "description": "Maximum number of valid"
                                          " time-series values"
                                          " to be returned.",
                           "schema": {
                               "type": "integer",
                           }
                       },
                   ])
    async def post(self, datasetId: str, varName: str):
        geo_json_object = self.request.json
        agg_methods = self.request.get_query_arg('aggMethods',
                                                 type=str,
                                                 default=None)
        agg_methods = agg_methods.split(',') if agg_methods else None
        start_date = self.request.get_query_arg('startDate',
                                                type=pd.Timestamp,
                                                default=None)
        end_date = self.request.get_query_arg('endDate',
                                              type=pd.Timestamp,
                                              default=None)
        max_valids = self.request.get_query_arg('maxValids',
                                                type=int,
                                                default=None)
        result = await self.ctx.run_in_executor(None,
                                                get_time_series,
                                                self.service_context,
                                                datasetId,
                                                varName,
                                                geo_json_object,
                                                agg_methods,
                                                start_date,
                                                end_date,
                                                max_valids)
        self.set_header('Content-Type', 'application/json')
        await self.finish(dict(result=result))
