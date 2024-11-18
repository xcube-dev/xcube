# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from xcube.util.assertions import assert_true
from xcube.util.cmaps import DEFAULT_CMAP_NAME
from .api import api
from .context import DatasetsContext
from .controllers import find_dataset_places
from .controllers import get_color_bars
from .controllers import get_dataset
from .controllers import get_dataset_coordinates
from .controllers import get_dataset_place_group
from .controllers import get_datasets
from .controllers import get_legend
from ..places import PATH_PARAM_PLACE_GROUP_ID

PATH_PARAM_DATASET_ID = {
    "name": "datasetId",
    "in": "path",
    "description": "Dataset identifier",
    "schema": {"type": "string"},
}

PATH_PARAM_VAR_NAME = {
    "name": "varName",
    "in": "path",
    "description": "Variable name",
    "schema": {"type": "string"},
}

QUERY_PARAM_CRS = {
    "name": "crs",
    "in": "query",
    "description": "The tile grid's spatial CRS",
    "schema": {"type": "string", "enum": ["EPSG:3857", "CRS84"], "default": "CRS84"},
}

QUERY_PARAM_VMIN = {
    "name": "vmin",
    "in": "query",
    "description": "Minimum value of variable" " for color mapping",
    "schema": {"type": "number", "default": 0},
}

QUERY_PARAM_VMAX = {
    "name": "vmax",
    "in": "query",
    "description": "Maximum value of variable" " for color mapping",
    "schema": {"type": "number", "default": 1},
}

QUERY_PARAM_CMAP = {
    "name": "cmap",
    "in": "query",
    "description": "Name of the (matplotlib) color mapping",
    "schema": {"type": "string", "default": DEFAULT_CMAP_NAME},
}

QUERY_PARAM_NORM = {
    "name": "norm",
    "in": "query",
    "description": "Name of the data normalisation applied before color mapping",
    "schema": {"enum": ["lin", "log", "cat"], "default": "lin"},
}


@api.route("/datasets")
class DatasetsHandler(ApiHandler[DatasetsContext]):
    """List the published datasets."""

    @api.operation(
        operation_id="getDatasets",
        summary="Get all datasets.",
        parameters=[
            {
                "name": "details",
                "in": "query",
                "description": "Whether to load dataset details",
                "schema": {
                    "type": "string",
                    "enum": ["0", "1"],
                },
            },
            {
                "name": "point",
                "in": "query",
                "description": "Only get datasets intersecting"
                ' given point. Format is "lon,lat",'
                ' for example "11.2,52.3"',
                "schema": {
                    "type": "string",
                },
            },
        ],
    )
    def get(self):
        granted_scopes = self.ctx.auth_ctx.get_granted_scopes(self.request.headers)
        details = self.request.get_query_arg("details", default=False)
        point = self.request.get_query_arg("point", default=None)
        if isinstance(point, str):
            try:
                point = tuple(map(float, point.split(",")))
                assert_true(len(point) == 2, "must be pair of two floats")
            except (ValueError, TypeError) as e:
                raise ApiError.BadRequest(f"illegal point: {e}")
        response = get_datasets(
            self.ctx,
            details=details,
            point=point,
            base_url=self.request.reverse_base_url,
            granted_scopes=granted_scopes,
        )
        self.response.finish(response)


@api.route("/datasets/{datasetId}")
class DatasetHandler(ApiHandler[DatasetsContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operation_id="getDataset",
        summary="Get the details of a dataset.",
        parameters=[PATH_PARAM_DATASET_ID],
    )
    async def get(self, datasetId: str):
        granted_scopes = self.ctx.auth_ctx.get_granted_scopes(self.request.headers)
        result = get_dataset(
            self.ctx,
            datasetId,
            base_url=self.request.reverse_base_url,
            granted_scopes=granted_scopes,
        )
        self.response.set_header("Content-Type", "application/json")
        await self.response.finish(result)


@api.route("/datasets/{datasetId}/coords/{dimName}")
class DatasetCoordsHandler(ApiHandler[DatasetsContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operation_id="getDatasetCoordinates",
        summary="Get the coordinates for a dimension of a dataset.",
        parameters=[PATH_PARAM_DATASET_ID],
    )
    async def get(self, datasetId: str, dimName: str):
        result = get_dataset_coordinates(self.ctx, datasetId, dimName)
        self.response.set_header("Content-Type", "application/json")
        await self.response.finish(result)


# noinspection PyPep8Naming
@api.route("/datasets/{datasetId}/places/{placeGroupId}")
class DatasetPlaceGroupHandler(ApiHandler[DatasetsContext]):
    """Get places for given dataset and place group."""

    @api.operation(
        operation_id="getDatasetPlaceGroup",
        summary="Get places for given dataset and place group.",
        parameters=[PATH_PARAM_DATASET_ID, PATH_PARAM_PLACE_GROUP_ID],
    )
    def get(self, datasetId: str, placeGroupId: str):
        response = get_dataset_place_group(
            self.ctx, datasetId, placeGroupId, self.request.reverse_base_url
        )
        self.response.finish(response)


# noinspection PyPep8Naming
@api.route("/places/{placeGroupId}/{datasetId}")
class PlacesForDatasetHandler(ApiHandler[DatasetsContext]):
    @api.operation(
        operation_id="findPlacesForDataset",
        tags=["places"],
        summary="Find places in place group for" " bounding box of given dataset.",
        parameters=[PATH_PARAM_PLACE_GROUP_ID, PATH_PARAM_DATASET_ID],
    )
    def get(self, placeGroupId: str, datasetId: str):
        query_expr = self.request.get_query_arg("query", default=None)
        comb_op = self.request.get_query_arg("comb", default="and")
        response = find_dataset_places(
            self.ctx,
            placeGroupId,
            datasetId,
            self.request.reverse_base_url,
            query_expr=query_expr,
            comb_op=comb_op,
        )
        self.response.finish(response)


# noinspection PyPep8Naming
class LegendHandler(ApiHandler[DatasetsContext]):
    async def get(self, datasetId: str, varName: str):
        legend = await self.ctx.run_in_executor(
            None,
            get_legend,
            self.ctx,
            datasetId,
            varName,
            {k: v[0] for k, v in self.request.query.items()},
        )
        self.response.set_header("Content-Type", "image/png")
        await self.response.finish(legend)


LEGEND_PARAMETERS = [
    PATH_PARAM_DATASET_ID,
    PATH_PARAM_VAR_NAME,
    QUERY_PARAM_CMAP,
    QUERY_PARAM_VMIN,
    QUERY_PARAM_VMAX,
    {
        "name": "width",
        "in": "query",
        "description": "Width of the legend in pixels",
        "schema": {"type": "number", "default": 256},
    },
    {
        "name": "height",
        "in": "query",
        "description": "Height of the legend in pixels",
        "schema": {"type": "number", "default": 16},
    },
]


# noinspection PyPep8Naming
@api.route("/datasets/{datasetId}/vars/{varName}/legend.png")
class OldLegendHandler(LegendHandler):
    @api.operation(
        operation_id="getLegendForVariable",
        summary="Get the legend as PNG used for the tiles"
        " for given variable. Deprecated!",
        parameters=LEGEND_PARAMETERS,
    )
    async def get(self, datasetId: str, varName: str):
        await super().get(datasetId, varName)


# noinspection PyPep8Naming
@api.route("/tiles/{datasetId}/{varName}/legend")
class NewLegendHandler(LegendHandler):
    @api.operation(
        operation_id="getLegendForTiles",
        tags=["tiles"],
        summary="Get the legend as PNG used for the tiles" " for given variable.",
        parameters=LEGEND_PARAMETERS,
    )
    async def get(self, datasetId: str, varName: str):
        await super().get(datasetId, varName)


# TODO (forman): move as endpoint "styles/colorbars" into API "styles"


@api.route("/colorbars")
class StylesColorBarsHandler(ApiHandler):
    """Get available color bars."""

    @api.operation(
        operation_id="getColorBars",
        summary="Get available color bars.",
        tags=["styles"],
    )
    def get(self):
        response = get_color_bars(self.ctx, "application/json")
        self.response.finish(response)


@api.route("/colorbars.html")
class StylesColorBarsHtmlHandler(ApiHandler):
    """Show available color bars."""

    @api.operation(
        operation_id="showColorBars",
        summary="Show available color bars.",
        tags=["styles"],
    )
    def get(self):
        response = get_color_bars(self.ctx, "text/html")
        self.response.finish(response)
