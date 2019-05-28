# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, Union, Sequence, Any, Optional, Collection

import numpy as np
import shapely.geometry
import xarray as xr

from ..api.select import select_vars
from ..util.geomask import where_geometry
from ..webapi.utils import get_dataset_geometry, GeoJSON


def get_time_series_for_point(dataset: xr.Dataset,
                              point: Union[shapely.geometry.Point, Dict[str, Any]],
                              var_names: Collection[str] = None,
                              start_date: np.datetime64 = None,
                              end_date: np.datetime64 = None) -> Optional[xr.Dataset]:
    if isinstance(point, dict):
        if not GeoJSON.is_point(point):
            raise ValueError("Invalid GeoJSON point")
        point = shapely.geometry.shape(point)

    bounds = get_dataset_geometry(dataset)
    if not bounds.contains(point):
        return None

    dataset = select_vars(dataset, var_names)
    if len(dataset.data_vars) == 0:
        return None

    if start_date is not None or end_date is not None:
        # noinspection PyTypeChecker
        dataset = dataset.sel(time=slice(start_date, end_date))

    dataset = dataset.sel(lon=point.x, lat=point.y, method='Nearest')
    return dataset.assign_attrs(max_number_of_observations=1)


def get_time_series_for_geometry(dataset: xr.Dataset,
                                 geometry: Union[shapely.geometry.base.BaseGeometry, Dict[str, Any]],
                                 var_names: Sequence[str] = None,
                                 use_groupby: bool = False,
                                 start_date: np.datetime64 = None,
                                 end_date: np.datetime64 = None) -> Optional[xr.Dataset]:
    if isinstance(geometry, dict):
        if not GeoJSON.is_geometry(geometry):
            raise ValueError("Invalid GeoJSON geometry")
        geometry = shapely.geometry.shape(geometry)

    if isinstance(geometry, shapely.geometry.Point):
        return get_time_series_for_point(dataset,
                                         geometry,
                                         var_names=var_names,
                                         start_date=start_date, end_date=end_date)

    if start_date is not None or end_date is not None:
        # noinspection PyTypeChecker
        dataset = dataset.sel(time=slice(start_date, end_date))

    dataset = where_geometry(dataset,
                             geometry,
                             var_names=var_names,
                             mask_var_name='__mask__')
    if dataset is None:
        return None

    mask = dataset['__mask__']
    dataset = dataset.drop('__mask__')

    if use_groupby:
        time_group = dataset.groupby('time')
        ds_mean = time_group.mean()
        ds_stdev = time_group.std()
        ds_count = time_group.count()
    else:
        ds_mean = dataset.mean(dim=('lat', 'lon'))
        ds_stdev = dataset.std(dim=('lat', 'lon'))
        ds_count = dataset.count(dim=('lat', 'lon'))

    ds_stdev = ds_stdev.rename(name_dict=dict({v: f"{v}_stdev" for v in ds_stdev.data_vars}))
    ds_count = ds_count.rename(name_dict=dict({v: f"{v}_count" for v in ds_count.data_vars}))
    dataset = xr.merge([ds_mean, ds_stdev, ds_count])
    dataset = dataset.assign_attrs(max_number_of_observations=np.count_nonzero(mask))

    return dataset
