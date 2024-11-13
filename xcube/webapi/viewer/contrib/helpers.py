from xcube.webapi.datasets.context import DatasetsContext
from xcube.server.api import Context


def get_datasets_ctx(ctx: Context) -> DatasetsContext:
    return ctx.get_api_ctx("datasets")
