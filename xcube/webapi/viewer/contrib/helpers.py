import xarray as xr

from xcube.webapi.datasets.context import DatasetsContext
from xcube.server.api import Context


def get_datasets_ctx(ctx: Context) -> DatasetsContext:
    return ctx.get_api_ctx("datasets")


def get_dataset(ctx: Context, dataset_id: str) -> xr.Dataset:
    return get_datasets_ctx(ctx).get_dataset(dataset_id)
