# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr

from xcube.server.api import Context
from xcube.webapi.datasets.context import DatasetsContext


def get_datasets_ctx(ctx: Context) -> DatasetsContext:
    return ctx.get_api_ctx("datasets")


def get_dataset(ctx: Context, dataset_id: str | None = None) -> xr.Dataset | None:
    return get_datasets_ctx(ctx).get_dataset(dataset_id) if dataset_id else None
