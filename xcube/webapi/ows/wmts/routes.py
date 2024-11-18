# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from .api import api
from .context import WmtsContext
from .controllers import WMTS_CRS84_TMS_ID
from .controllers import WMTS_TILE_FORMAT
from .controllers import WMTS_VERSION
from .controllers import WMTS_WEB_MERCATOR_TMS_ID
from .controllers import get_crs_name_from_tms_id
from .controllers import get_wmts_capabilities_xml
from ...datasets import PATH_PARAM_DATASET_ID
from ...datasets import PATH_PARAM_VAR_NAME
from ...datasets import QUERY_PARAM_CMAP
from ...datasets import QUERY_PARAM_CRS
from ...datasets import QUERY_PARAM_NORM
from ...datasets import QUERY_PARAM_VMAX
from ...datasets import QUERY_PARAM_VMIN
from ...tiles import PATH_PARAM_X
from ...tiles import PATH_PARAM_Y
from ...tiles import PATH_PARAM_Z
from ...tiles import QUERY_PARAM_RETINA
from ...tiles import QUERY_PARAM_TIME
from ...tiles.controllers import compute_ml_dataset_tile

_VALID_WMTS_TMS_IDS = (WMTS_CRS84_TMS_ID, WMTS_WEB_MERCATOR_TMS_ID)

PATH_PARAM_TMS_ID = {
    "name": "tmsId",
    "in": "query",
    "description": "Tile matrix set identifier",
    "schema": {
        "type": "string",
        "enum": [WMTS_CRS84_TMS_ID, WMTS_WEB_MERCATOR_TMS_ID],
    },
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
    QUERY_PARAM_RETINA,
]

TMS_TILE_PARAMETERS = [
    PATH_PARAM_DATASET_ID,
    PATH_PARAM_VAR_NAME,
    PATH_PARAM_Z,
    PATH_PARAM_Y,
    PATH_PARAM_X,
    QUERY_PARAM_CRS,
    QUERY_PARAM_NORM,
    QUERY_PARAM_VMIN,
    QUERY_PARAM_VMAX,
    QUERY_PARAM_CMAP,
    QUERY_PARAM_TIME,
    QUERY_PARAM_RETINA,
]


@api.route("/wmts/1.0.0/WMTSCapabilities.xml")
class WmtsCapabilitiesXmlHandler(ApiHandler[WmtsContext]):
    @api.operation(
        operationId="getWmtsCapabilities",
        summary="Gets the WMTS capabilities as XML document",
    )
    async def get(self):
        self.request.make_query_lower_case()
        capabilities = await self.ctx.run_in_executor(
            None,
            get_wmts_capabilities_xml,
            self.ctx,
            self.request.reverse_base_url,
            WMTS_CRS84_TMS_ID,
        )
        self.response.set_header("Content-Type", "application/xml")
        await self.response.finish(capabilities)


@api.route("/wmts/1.0.0/{tmsId}/WMTSCapabilities.xml")
class WmtsCapabilitiesXmlForTmsHandler(ApiHandler[WmtsContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operationId="getWmtsTmsCapabilities",
        summary="Gets the WMTS capabilities" " for tile matrix set as XML document",
        parameters=[PATH_PARAM_TMS_ID],
    )
    async def get(self, tmsId):
        self.request.make_query_lower_case()
        _assert_valid_tms_id(tmsId)
        capabilities = await self.ctx.run_in_executor(
            None,
            get_wmts_capabilities_xml,
            self.ctx,
            self.request.reverse_base_url,
            tmsId,
        )
        self.response.set_header("Content-Type", "application/xml")
        await self.response.finish(capabilities)


@api.route("/wmts/1.0.0/tile/{datasetId}/{varName}/{z}/{y}/{x}.png")
class WmtsImageTileHandler(ApiHandler[WmtsContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operationId="getWmtsImageTile",
        summary="Gets a WMTS image tile in PNG format.",
        parameters=TILE_PARAMETERS,
    )
    async def get(self, datasetId: str, varName: str, z: str, y: str, x: str):
        self.request.make_query_lower_case()
        tms_id = self.request.get_query_arg("tilematrixset", default=WMTS_CRS84_TMS_ID)
        _assert_valid_tms_id(tms_id)
        crs_name = get_crs_name_from_tms_id(tms_id)
        tile = await self.ctx.run_in_executor(
            None,
            compute_ml_dataset_tile,
            self.ctx.tiles_ctx,
            datasetId,
            varName,
            crs_name,
            x,
            y,
            z,
            _query_to_dict(self.request),
        )
        self.response.set_header("Content-Type", "image/png")
        await self.response.finish(tile)


@api.route("/wmts/1.0.0/tile/{datasetId}/{varName}/{tmsId}/{z}/{y}/{x}.png")
class WmtsImageTileForTmsHandler(ApiHandler[WmtsContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operationId="getWmtsTmsImageTile",
        summary="Gets a WMTS image tile" " for given tile matrix set in PNG format",
        parameters=TMS_TILE_PARAMETERS,
    )
    async def get(
        self, datasetId: str, varName: str, tmsId: str, z: str, y: str, x: str
    ):
        self.request.make_query_lower_case()
        _assert_valid_tms_id(tmsId)
        crs_name = get_crs_name_from_tms_id(tmsId)
        tile = await self.ctx.run_in_executor(
            None,
            compute_ml_dataset_tile,
            self.ctx.tiles_ctx,
            datasetId,
            varName,
            crs_name,
            x,
            y,
            z,
            _query_to_dict(self.request),
        )
        self.response.set_header("Content-Type", "image/png")
        await self.response.finish(tile)


@api.route("/wmts/kvp")
class WmtsKvpHandler(ApiHandler[WmtsContext]):
    @api.operation(
        operationId="invokeWmtsMethodFromKvp",
        summary="Invokes the WMTS by key-value pairs",
        parameters=TILE_PARAMETERS,
    )
    async def get(self):
        self.request.make_query_lower_case()
        service = self.request.get_query_arg("service")
        if service != "WMTS":
            raise ApiError.BadRequest('value for "service" parameter must be "WMTS"')
        request = self.request.get_query_arg("request")
        if request == "GetCapabilities":
            wmts_version = self.request.get_query_arg("version", default=WMTS_VERSION)
            if wmts_version != WMTS_VERSION:
                raise ApiError.BadRequest(
                    f'value for "version" parameter must be "{WMTS_VERSION}"'
                )
            tms_id = self.request.get_query_arg(
                "tilematrixset", default=WMTS_CRS84_TMS_ID
            )
            _assert_valid_tms_id(tms_id)
            capabilities = await self.ctx.run_in_executor(
                None,
                get_wmts_capabilities_xml,
                self.ctx,
                self.request.reverse_base_url,
                tms_id,
            )
            self.response.set_header("Content-Type", "application/xml")
            await self.response.finish(capabilities)

        elif request == "GetTile":
            wmts_version = self.request.get_query_arg("version", default=WMTS_VERSION)
            if wmts_version != WMTS_VERSION:
                raise ApiError.BadRequest(
                    f'value for "version" parameter must be "{WMTS_VERSION}"'
                )
            layer = self.request.get_query_arg("layer")
            try:
                ds_id, var_name = layer.split(".")
            except ValueError as e:
                raise ApiError.BadRequest(
                    'value for "layer" parameter must be' ' "<dataset>.<variable>"'
                ) from e
            # For time being, we ignore "style"
            # style = self.request.get_query_arg("style"
            mime_type = self.request.get_query_arg(
                "format", default=WMTS_TILE_FORMAT
            ).lower()
            if mime_type not in (WMTS_TILE_FORMAT, "png"):
                raise ApiError.BadRequest(
                    f'value for "format" parameter' f' must be "{WMTS_TILE_FORMAT}"'
                )
            tms_id = self.request.get_query_arg(
                "tilematrixset", default=WMTS_CRS84_TMS_ID
            )
            _assert_valid_tms_id(tms_id)
            crs_name = get_crs_name_from_tms_id(tms_id)
            x = self.request.get_query_arg("tilecol", type=int)
            y = self.request.get_query_arg("tilerow", type=int)
            z = self.request.get_query_arg("tilematrix", type=int)
            tile = await self.ctx.run_in_executor(
                None,
                compute_ml_dataset_tile,
                self.ctx.tiles_ctx,
                ds_id,
                var_name,
                crs_name,
                x,
                y,
                z,
                _query_to_dict(self.request),
            )
            self.response.set_header("Content-Type", "image/png")
            await self.response.finish(tile)
        elif request == "GetFeatureInfo":
            raise ApiError.BadRequest(
                'request type "GetFeatureInfo" not yet implemented'
            )
        else:
            raise ApiError.BadRequest(f'invalid request type "{request}"')


def _assert_valid_tms_id(tms_id: str):
    if tms_id not in _VALID_WMTS_TMS_IDS:
        raise ApiError.BadRequest(
            f'value for "tilematrixset" parameter'
            f" must be one of {_VALID_WMTS_TMS_IDS!r}"
        )


def _query_to_dict(request):
    return {k: v[0] for k, v in request.query.items()}
