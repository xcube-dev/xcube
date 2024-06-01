# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiHandler
from .api import api
from .context import StatisticsContext
from .controllers import compute_statistics


# noinspection PyPep8Naming
@api.route("/statistics/{datasetId}/{varName}")
class StatisticsHandler(ApiHandler[StatisticsContext]):
    @api.operation(
        operation_id="getStatistics",
        summary="Get statistics for a given dataset variable.",
    )
    async def post(self, datasetId: str, varName: str):
        result = await self.ctx.run_in_executor(
            None,
            compute_statistics,
            self.ctx,
            datasetId,
            varName,
            self.request.json,
            {k: v[0] for k, v in self.request.query.items()},
        )
        await self.response.finish({"result": result})
