# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiError, ApiHandler

from .api import api
from .context import PlacesContext
from .controllers import find_places

PATH_PARAM_PLACE_GROUP_ID = {
    "name": "placeGroupId",
    "in": "path",
    "description": "Place group identifier",
    "schema": {"type": "string"},
}


@api.route("/places")
class PlaceGroupsHandler(ApiHandler[PlacesContext]):
    @api.operation(
        operationId="getPlaceGroups", summary="Get place groups including all places."
    )
    def get(self):
        place_groups = self.ctx.get_global_place_groups(self.request.reverse_base_url)
        self.response.finish({"placeGroups": place_groups})


# noinspection PyPep8Naming
@api.route("/places/{placeGroupId}")
class FindPlacesHandler(ApiHandler):
    """Find places within a known place group."""

    @api.operation(
        operationId="findPlacesInPlaceGroup",
        summary="Find places in a given place group.",
        parameters=[
            PATH_PARAM_PLACE_GROUP_ID,
            {
                "name": "geom",
                "in": "query",
                "description": "Geometry WKT string",
                "schema": {"type": "string"},
            },
            {
                "name": "bbox",
                "in": "query",
                "description": "Bounding box string in the form x1,y1,x2,y2",
                "schema": {"type": "string"},
            },
        ],
    )
    def get(self, placeGroupId: str):
        geom_wkt = self.request.get_query_arg("geom", default=None)
        box_coords = self.request.get_query_arg("bbox", default=None)
        # Not implemented yet:
        # query_expr = self.request.get_query_arg("query", default=None)
        # comb_op = self.request.get_query_arg("comb", default="and")
        if geom_wkt and box_coords:
            raise ApiError.BadRequest('Only one of "geom" and "bbox" may be given')
        places = find_places(
            self.ctx,
            placeGroupId,
            self.request.reverse_base_url,
            query_geometry=box_coords or geom_wkt or None,
            # query_expr=query_expr,
            # comb_op=comb_op
        )
        self.response.finish({"places": places})

    @api.operation(
        operationId="findPlacesInPlaceGroup",
        summary="Find places in a given place group for a GeoJSON object.",
        parameters=[PATH_PARAM_PLACE_GROUP_ID],
        description="The request body must be a GeoJSON object.",
    )
    def post(self, placeGroupId: str):
        # Not implemented yet:
        # query_expr = self.request.get_query_arg("query", default=None)
        # comb_op = self.request.get_query_arg("comb", default="and")
        geojson_obj = self.request.json
        places = find_places(
            self.ctx,
            placeGroupId,
            self.request.reverse_base_url,
            query_geometry=geojson_obj,
            # query_expr=query_expr,
            # comb_op=comb_op
        )
        self.response.finish({"places": places})
