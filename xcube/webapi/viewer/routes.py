# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

import importlib.resources
import os
import pathlib
from typing import Union, Optional

from xcube.constants import LOG
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from .api import api
from .context import ViewerContext

ENV_VAR_XCUBE_VIEWER_PATH = "XCUBE_VIEWER_PATH"


@api.route('/viewer/config/{*path}')
class ViewerConfigHandler(ApiHandler[ViewerContext]):
    @api.operation(operationId='getViewerConfigurationItem',
                   summary='Get a configuration item for the xcube viewer.')
    def get(self, path: Optional[str]):
        config_items = self.ctx.config_items
        if config_items is None:
            raise ApiError.NotFound(f'xcube viewer has'
                                    f' not been been configured')
        try:
            data = config_items[path]
        except KeyError:
            raise ApiError.NotFound(f'The item {path!r} was not found'
                                    f' in viewer configuration')
        content_type = self.get_content_type(path)
        self.response.set_header("Content-Length", str(len(data)))
        if content_type is not None:
            self.response.set_header("Content-Type", content_type)
        self.response.write(data)

    @staticmethod
    def get_content_type(path: str) -> Optional[str]:
        filename_ext = path.split('/')[-1].split('.')[-1]
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


_responses = {
    200: {
        "description": "OK",
        "content": {
            "text/html": {
                "schema": {
                    "type": "string"
                },
            },
        }
    }
}

_viewer_module = "xcube.webapi.viewer"
_data_dir = "data"
_default_filename = "index.html"


@api.static_route('/viewer',
                  default_filename=_default_filename,
                  summary="Brings up the xcube Viewer webpage",
                  responses=_responses)
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
    LOG.warning(f"Cannot find {_data_dir}/{_default_filename}"
                f" in {_viewer_module}, consider setting environment variable"
                f" {ENV_VAR_XCUBE_VIEWER_PATH}",
                exc_info=True)
    return None
