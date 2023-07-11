from typing import Tuple

import xarray as xr

from xcube.core.gridmapping import GridMapping
from .op import op


@op()
def spatial_subset(dataset: xr.Dataset,
                   bbox: Tuple[float, float, float, float]) -> xr.Dataset:
    x1, y1, x2, y2 = bbox
    gm = GridMapping.from_dataset(dataset)
    x_name, y_name = gm.xy_dim_names
    return dataset.sel({
        x_name: slice(x1, x2),
        y_name: slice(y1, y2) if gm.is_j_axis_up else slice(y2, y1)
    })
