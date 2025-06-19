# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
from typing import Any

import xarray as xr

from xcube.server.api import Context
from xcube.webapi.datasets.context import DatasetsContext


def get_datasets_ctx(ctx: Context) -> DatasetsContext:
    return ctx.get_api_ctx("datasets")


def get_dataset(ctx: Context, dataset_id: str | None = None) -> xr.Dataset | None:
    return get_datasets_ctx(ctx).get_dataset(dataset_id) if dataset_id else None


def get_place_label(place_id: str, place_group: list[dict[str, Any]]) -> str | None:
    if not place_group or not place_id:
        return None
    for place in place_group:
        for features in place["features"]:
            if features["id"] == place_id:
                return features["properties"]["label"]
    return None
