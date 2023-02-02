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
from typing import Union

from xcube.constants import LOG
from .api import api

ENV_VAR_XCUBE_VIEWER_PATH = "XCUBE_VIEWER_PATH"

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
