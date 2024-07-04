# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os
from typing import Any

from xcube.util.versions import get_xcube_versions
from xcube.version import version
from .context import MetaContext


def get_service_info(ctx: MetaContext) -> dict[str, Any]:
    api_infos = []
    for other_api in ctx.apis:
        api_info = {
            "name": other_api.name,
            "version": other_api.version,
            "description": other_api.description,
        }
        api_infos.append({k: v for k, v in api_info.items() if v is not None})

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
        pid=os.getpid(),
    )
