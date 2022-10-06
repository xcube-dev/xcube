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

import os
from typing import Any, Dict

from xcube.util.versions import get_xcube_versions
from xcube.version import version
from .context import MetaContext


def get_service_info(ctx: MetaContext) -> Dict[str, Any]:
    api_infos = []
    for other_api in ctx.apis:
        api_info = {
            "name": other_api.name,
            "version": other_api.version,
            "description": other_api.description
        }
        api_infos.append({k: v
                          for k, v in api_info.items()
                          if v is not None})

    # TODO (forman): once APIs are configurable, we should
    #   get the server name, description from configuration too
    return dict(
        name="xcube Server",
        description="xcube Server",
        version=version,
        versions=get_xcube_versions(),
        apis=api_infos,
        startTime=ctx.start_time,
        currentTime=ctx.current_time,
        updateTime=ctx.update_time,
        pid=os.getpid()
    )
