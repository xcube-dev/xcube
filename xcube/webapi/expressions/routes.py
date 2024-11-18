# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiHandler
from ..datasets.routes import PATH_PARAM_DATASET_ID
from .api import api
from .context import ExpressionsContext
from .controllers import get_expressions_capabilities
from .controllers import validate_expression

PATH_PARAM_VAR_EXPR = {
    "name": "varExpr",
    "in": "path",
    "description": "Variable expression",
    "schema": {"type": "string"},
}


@api.route("/expressions/capabilities")
class ExpressionsNamespaceHandler(ApiHandler[ExpressionsContext]):
    @api.operation(
        operation_id="getExpressionCapabilities",
        summary=(
            "Gets the server capabilities for expressions that define user variables"
        ),
    )
    def get(self):
        self.response.finish(get_expressions_capabilities(self.ctx))


# noinspection PyPep8Naming
@api.route("/expressions/validate/{datasetId}/{varExpr}")
class ExpressionsNamespaceHandler(ApiHandler[ExpressionsContext]):
    @api.operation(
        operation_id="validateExpression",
        summary="Validate the given variable expression",
        parameters=[PATH_PARAM_DATASET_ID, PATH_PARAM_VAR_EXPR],
    )
    def get(self, datasetId: str, varExpr: str):
        self.response.finish(validate_expression(self.ctx, datasetId, varExpr))
