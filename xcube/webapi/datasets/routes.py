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

from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from .api import api
from .context import DatasetsContext
from .controllers import find_dataset_places
from .controllers import get_datasets
from ...util.assertions import assert_true


@api.route('/datasets')
class DatasetsHandler(ApiHandler[DatasetsContext]):
    """List the published datasets."""

    @api.operation(
        operation_id='getDatasets',
        parameters=[
            {
                "name": "details",
                "in": "query",
                "description": "Whether to load dataset details",
                "schema": {
                    "type": "string",
                    "enum": ["0", "1"],
                }
            },
            {
                "name": "point",
                "in": "query",
                "description": "Only get datasets intersecting"
                               " given point. Format is \"lon,lat\","
                               " for example \"11.2,52.3\"",
                "schema": {
                    "type": "string",
                },
            },
        ]
    )
    def get(self):
        granted_scopes = self.ctx.auth_ctx.granted_scopes(
            self.request.headers)
        details = self.request.get_query_arg('details', default=False)
        point = self.request.get_query_arg('point', None)
        if point is not None:
            try:
                point = tuple(map(float, point.split(',')))
                assert_true(len(point) == 2, 'must be pair of two floats')
            except (ValueError, TypeError) as e:
                raise ApiError.BadRequest(f"illegal point: {e}")
        response = get_datasets(self.ctx,
                                details=details,
                                point=point,
                                base_url=self.request.base_url,
                                granted_scopes=granted_scopes)
        self.response.finish(response)


@api.route('/datasets/{ds_id}/{place_group_id}')
class DatasetPlacesHandler(ApiHandler):
    """Get places for given dataset and place group."""

    @api.operation(operation_id='getDatasetPlaces')
    def get(self, ds_id: str, place_group_id: str):
        query_expr = self.request.get_query_arg("query", default=None)
        comb_op = self.request.get_query_arg("comb", default="and")
        response = find_dataset_places(self.service_context,
                                       place_group_id,
                                       ds_id,
                                       self.base_url,
                                       query_expr=query_expr,
                                       comb_op=comb_op)
        self.response.finish(response)


@api.route('/places/{place_group_id}/{ds_id}')
class PlacesForDatasetHandler(DatasetPlacesHandler):
    @api.operation(operation_id='getPlacesForDataset')
    def get(self, place_group_id: str, ds_id: str):
        return super().get(ds_id, place_group_id)
