# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr

from xcube.server.api import Context
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.datasets.controllers import (
    get_dataset_title_and_description as _get_dataset_title_and_description,
)


def get_datasets_ctx(ctx: Context) -> DatasetsContext:
    return ctx.get_api_ctx("datasets")


def get_dataset(ctx: Context, dataset_id: str | None = None) -> xr.Dataset | None:
    return get_datasets_ctx(ctx).get_dataset(dataset_id) if dataset_id else None


def get_dataset_title_and_description(
    ctx: Context, dataset_id: str | None = None
) -> tuple[str | None, str | None]:
    if dataset_id:
        dataset = get_datasets_ctx(ctx).get_dataset(dataset_id)
        dataset_config = get_datasets_ctx(ctx).get_dataset_config(dataset_id)
        return _get_dataset_title_and_description(dataset, dataset_config)
    return None, None
