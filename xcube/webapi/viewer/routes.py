# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import importlib.resources
import os
import pathlib
from typing import Union, Optional

from dashipy import __version__ as dashi_version
from dashipy.controllers import get_callback_results
from dashipy.controllers import get_contributions
from dashipy.controllers import get_layout

from xcube.constants import LOG
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from .api import api
from .context import ViewerContext


ENV_VAR_XCUBE_VIEWER_PATH = "XCUBE_VIEWER_PATH"

_viewer_module = "xcube.webapi.viewer"
_data_dir = "dist"
_default_filename = "index.html"

_responses = {
    200: {
        "description": "OK",
        "content": {
            "text/html": {
                "schema": {"type": "string"},
            },
        },
    }
}


@api.static_route(
    "/viewer",
    default_filename=_default_filename,
    summary="Brings up the xcube Viewer webpage",
    responses=_responses,
)
def get_local_viewer_path() -> Union[None, str, pathlib.Path]:
    local_viewer_path = os.environ.get(ENV_VAR_XCUBE_VIEWER_PATH)
    if local_viewer_path:
        return local_viewer_path
    try:
        with importlib.resources.files(_viewer_module) as path:
            local_viewer_path = path / _data_dir
            if (local_viewer_path / _default_filename).is_file():
                return local_viewer_path
    except ImportError:
        pass
    LOG.warning(
        f"Cannot find {_data_dir}/{_default_filename}"
        f" in {_viewer_module}, consider setting environment variable"
        f" {ENV_VAR_XCUBE_VIEWER_PATH}",
        exc_info=True,
    )
    return None


@api.route("/viewer/config/{*path}")
class ViewerConfigHandler(ApiHandler[ViewerContext]):
    @api.operation(
        operationId="getViewerConfigurationItem",
        summary="Get a configuration item for the xcube viewer.",
    )
    def get(self, path: Optional[str]):
        config_items = self.ctx.config_items
        if config_items is None:
            raise ApiError.NotFound(f"xcube viewer has" f" not been been configured")
        try:
            data = config_items[path]
        except KeyError:
            raise ApiError.NotFound(
                f"The item {path!r} was not found" f" in viewer configuration"
            )
        content_type = self.get_content_type(path)
        self.response.set_header("Content-Length", str(len(data)))
        if content_type is not None:
            self.response.set_header("Content-Type", content_type)
        self.response.write(data)

    @staticmethod
    def get_content_type(path: str) -> Optional[str]:
        filename_ext = path.split("/")[-1].split(".")[-1]
        if filename_ext in ("json", "xml"):
            return f"application/{filename_ext}; charset=UTF-8"
        elif filename_ext in ("csv", "html", "css", "js"):
            return f"text/{filename_ext}; charset=UTF-8"
        elif filename_ext in ("geojson",):
            return "application/geo+json; charset=UTF-8"
        elif filename_ext in ("gif", "png"):
            return f"image/{filename_ext}"
        elif filename_ext in ("jpeg", "jpg"):
            return "image/jpeg"
        return None


@api.route("/viewer/ext")
class ViewerExtRootHandler(ApiHandler[ViewerContext]):
    # GET /
    def get(self):
        self.response.set_header("Content-Type", "text/plain")
        self.response.write(f"dashi-server {dashi_version}")


@api.route("/viewer/ext/contributions")
class ViewerExtContributionsHandler(ApiHandler[ViewerContext]):

    # GET /dashi/contributions
    @api.operation(
        operationId="getViewerExtContributions",
        summary="Get viewer extensions and all their contributions.",
    )
    def get(self):
        self.response.write(get_contributions(self.ctx))


# noinspection PyPep8Naming
@api.route("/viewer/ext/layout/{contribPoint}/{contribIndex}")
class ViewerExtLayoutHandler(ApiHandler[ViewerContext]):
    # GET /dashi/layout/{contrib_point_name}/{contrib_index}
    def get(self, contribPoint: str, contribIndex: str):
        self.response.write(get_layout(self.ctx, contribPoint, int(contribIndex), {}))

    # POST /dashi/layout/{contrib_point_name}/{contrib_index}
    @api.operation(
        operationId="getViewerExtLayout",
        summary="Get the initial layout for the given viewer contribution.",
        parameters=[
            {
                "name": "contribPoint",
                "in": "path",
                "description": 'Contribution point name, e.g., "panels"',
                "schema": {"type": "string"},
            },
            {
                "name": "contribIndex",
                "in": "path",
                "description": "Contribution index",
                "schema": {"type": "integer"},
            },
        ],
    )
    def post(self, contribPoint: str, contribIndex: str):
        self.response.write(
            get_layout(self.ctx, contribPoint, int(contribIndex), self.request.json)
        )


@api.route("/viewer/ext/callback")
class ViewerExtCallbackHandler(ApiHandler[ViewerContext]):

    # POST /dashi/callback
    @api.operation(
        operationId="invokeViewerCallbacks",
        summary="Process the viewer contribution callback requests"
        " and return state change requests.",
    )
    def post(self):
        self.response.write(get_callback_results(self.ctx, self.request.json))
