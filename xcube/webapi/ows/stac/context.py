# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


from xcube.server.api import Context
from xcube.webapi.common.context import ResourcesContext

from ...datasets.context import DatasetsContext


class StacContext(ResourcesContext):
    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self._datasets_ctx = server_ctx.get_api_ctx("datasets", cls=DatasetsContext)

        # check: determine what else to include in this list.
        #  Listing all 11396 CRSs currently known to pyproj seems
        #  impractical. Make it a configuration option, perhaps?
        self._available_crss = ["OGC:CRS84", "EPSG:4326"]

    @property
    def datasets_ctx(self) -> DatasetsContext:
        return self._datasets_ctx

    @property
    def available_crss(self) -> list[str]:
        return self._available_crss
