from typing import Collection, Optional, Tuple

import numpy as np
import xarray as xr
from xcube.core.geocoding import GeoCoding


def select_variables_subset(dataset: xr.Dataset, var_names: Collection[str] = None) -> xr.Dataset:
    """
    Select data variable from given *dataset* and create new dataset.

    :param dataset: The dataset from which to select variables.
    :param var_names: The names of data variables to select.
    :return: A new dataset. It is empty, if *var_names* is empty. It is *dataset*, if *var_names* is None.
    """
    if var_names is None:
        return dataset
    dropped_variables = set(dataset.data_vars.keys()).difference(var_names)
    if not dropped_variables:
        return dataset
    return dataset.drop_vars(dropped_variables)


def select_spatial_subset(dataset: xr.Dataset,
                          ij_bbox: Tuple[int, int, int, int] = None,
                          ij_border: int = 0,
                          xy_bbox: Tuple[float, float, float, float] = None,
                          xy_border: float = 0.,
                          geo_coding: GeoCoding = None,
                          xy_names: Tuple[str, str] = None) -> Optional[xr.Dataset]:
    """
    Select a spatial subset of *dataset* for the bounding box *bbox*.

    :param dataset: Source dataset.
    :param ij_bbox: Bounding box (i_min, i_min, j_max, j_max) in pixel coordinates.
    :param geo_coding: Optional dataset geo-coding.
    :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*. Ignored if *geo_coding* is given.
    :return: Spatial dataset subset
    """
    NORMAN_SOLUTION = True

    if NORMAN_SOLUTION:
        if ij_bbox is None and xy_bbox is None:
            raise ValueError('One of ij_bbox and xy_bbox must be given')
        if ij_bbox and xy_bbox:
            raise ValueError('Only one of ij_bbox and xy_bbox can be given')
        geo_coding = geo_coding if geo_coding is not None else GeoCoding.from_dataset(dataset, xy_names=xy_names)
        if xy_bbox:
            ij_bbox = geo_coding.ij_bbox(xy_bbox, ij_border=ij_border, xy_border=xy_border)
            if ij_bbox[0] == -1:
                return None
        width, height = geo_coding.size
        i_min, j_min, i_max, j_max = ij_bbox
        if i_min > 0 or j_min > 0 or i_max < width - 1 or j_max < height - 1:
            x_dim, y_dim = geo_coding.dims
            i_slice = slice(i_min, i_max + 1)
            j_slice = slice(j_min, j_max + 1)
            return dataset.isel({x_dim: i_slice, y_dim: j_slice})
        return dataset

    else:
        if ij_bbox is None and xy_bbox is None:
            raise ValueError('One of ij_bbox and xy_bbox must be given')
        if ij_bbox and xy_bbox:
            raise ValueError('Only one of ij_bbox and xy_bbox can be given')
        dataset_subset = dataset.copy()
        if xy_bbox:
            lon_min, lat_min, lon_max, lat_max = xy_bbox

            if lon_max <= dataset.lon.min() or lon_min >= dataset.lon.max() \
                    or lat_max <= dataset.lat.min() or lat_min >= dataset.lat.max():
                return None
            if lon_max < dataset.lon.min() or lon_min > dataset.lon.max() \
                    or lat_max < dataset.lat.min() or lat_min > dataset.lat.max():
                return None
            if lon_min < dataset.lon.min() and lat_min < dataset.lat.min() \
                    and lon_max > dataset.lon.max() and lat_max > dataset.lat.max():
                return dataset
            dataset_subset.coords['x'] = xr.DataArray(np.arange(0, dataset.x.size), dims='x')
            dataset_subset.coords['y'] = xr.DataArray(np.arange(0, dataset.y.size), dims='y')
            lon_subset = dataset_subset.lon.where((dataset_subset.lon >= lon_min) & (dataset_subset.lon <= lon_max),
                                                  drop=True)
            lat_subset = dataset_subset.lat.where((dataset_subset.lat >= lat_min) & (dataset_subset.lat <= lat_max),
                                                  drop=True)
            x1 = lon_subset.x[0]
            x2 = lon_subset.x[-1]
            y1 = lat_subset.y[0]
            y2 = lat_subset.y[-1]
            x1, y1, x2, y2 = tuple(map(int, (x1, y1, x2, y2)))
        if ij_bbox:
            x1, y1, x2, y2 = ij_bbox
            if x2 <= dataset.x.min() or x1 >= dataset.x.max() \
                    or y2 <= dataset.y.min() or y1 >= dataset.y.max():
                return None
            if x2 < dataset.x.min() or x1 > dataset.x.max() \
                    or y2 < dataset.y.min() or y1 > dataset.y.max():
                return None
            if x1 <= dataset.x.min() and y1 <= dataset.x.min() \
                    and x2 >= dataset.y.max() and y2 >= dataset.y.max():
                return dataset
            x1, y1, x2, y2 = ij_bbox
        dataset = dataset_subset.isel(x=slice(x1, x2 + 1), y=slice(y1, y2 + 1))
        return dataset
