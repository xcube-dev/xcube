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
        summary=(
            "Get statistics of a dataset variable for given time stamp and geometry."
        ),
        description=(
            "The geometry is passed in the request body in"
            " form of a valid GeoJSON geometry object."
            " The operation returns the count, minimum, maximum, mean,"
            " and standard deviation of a data variable. If a 2-D geometry"
            " is passed, a histogram is returned as well."
        ),
        parameters=[
            {
                "name": "time",
                "in": "query",
                "description": ('Timestamp using format "YYYY-MM-DD hh:mm:ss"'),
                "required": True,
                "schema": {"type": "string", "format": "datetime"},
            }
        ],
    )
    async def post(self, datasetId: str, varName: str):
        params = {k: v[0] for k, v in self.request.query.items()}
        result = await self.ctx.run_in_executor(
            None,
            compute_statistics,
            self.ctx,
            datasetId,
            varName,
            self.request.json,
            params,
        )
        await self.response.finish({"result": result})
