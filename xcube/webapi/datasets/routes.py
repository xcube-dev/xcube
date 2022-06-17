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
from .controllers import get_color_bars
from .controllers import get_dataset_place_group
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


# noinspection PyPep8Naming
@api.route('/datasets/{datasetId}/places/{placeGroupId}')
class DatasetPlaceGroupHandler(ApiHandler[DatasetsContext]):
    """Get places for given dataset and place group."""

    @api.operation(operation_id='getDatasetPlaceGroup',
                   summary='Get places for given dataset and place group')
    def get(self, datasetId: str, placeGroupId: str):
        response = get_dataset_place_group(self.ctx,
                                           datasetId,
                                           placeGroupId,
                                           self.request.base_url)
        self.response.finish(response)


# noinspection PyPep8Naming
@api.route('/places/{placeGroupId}/{datasetId}')
class PlacesForDatasetHandler(ApiHandler[DatasetsContext]):
    @api.operation(operation_id='findPlacesForDataset',
                   tags=['places'],
                   summary='Find places in place group for'
                           ' bounding box of given dataset')
    def get(self, placeGroupId: str, datasetId: str):
        query_expr = self.request.get_query_arg("query", default=None)
        comb_op = self.request.get_query_arg("comb", default="and")
        response = find_dataset_places(self.ctx,
                                       placeGroupId,
                                       datasetId,
                                       self.request.base_url,
                                       query_expr=query_expr,
                                       comb_op=comb_op)
        self.response.finish(response)


# TODO (forman): move as endpoint "styles/colorbars" into API "styles"

@api.route('/colorbars')
class StylesColorBarsHandler(ApiHandler):
    """Get available color bars."""

    @api.operation(operation_id='getColorBars',
                   summary="Get available color bars.",
                   tags=['styles'])
    def get(self):
        response = get_color_bars(self.ctx, 'application/json')
        self.response.finish(response)


@api.route('/colorbars.html')
class StylesColorBarsHtmlHandler(ApiHandler):
    """Show available color bars."""

    @api.operation(operation_id='getColorBars',
                   summary="Show available color bars.",
                   tags=['styles'])
    def get(self):
        response = get_color_bars(self.ctx, 'text/html')
        self.response.finish(response)
