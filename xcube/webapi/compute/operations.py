# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Tuple

import xarray as xr

from xcube.core.gridmapping import GridMapping

from xcube.webapi.compute.op.decorator import operation
from xcube.webapi.compute.op.decorator import op_param


@operation
@op_param(
    "bbox",
    title="Bounding box",
    description="Bounding box using the dataset's CRS coordinates",
    # The rest of the schema is inferred from the function signature.
)
def spatial_subset(
    dataset: xr.Dataset, bbox: tuple[float, float, float, float]
) -> xr.Dataset:
    """Create a spatial subset from given dataset."""
    x1, y1, x2, y2 = bbox
    gm = GridMapping.from_dataset(dataset)
    x_name, y_name = gm.xy_dim_names
    return dataset.sel(
        {
            x_name: slice(x1, x2),
            y_name: slice(y1, y2) if gm.is_j_axis_up else slice(y2, y1),
        }
    )
