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

from typing import Union, Sequence, Optional

import numpy as np
import shapely.geometry
import shapely.wkt
import xarray as xr

from ..api.select import select_vars
from ..api.verify import assert_cube
from ..util.geom import where_geometry, convert_geometry, Geometry, get_dataset_geometry

Date = Union[np.datetime64, str]


def get_time_series(cube: xr.Dataset,
                    geometry: Geometry = None,
                    var_names: Sequence[str] = None,
                    start_date: Date = None,
                    end_date: Date = None,
                    use_groupby: bool = False,
                    include_count: bool = False,
                    include_stdev: bool = False,
                    cube_asserted: bool = False) -> Optional[xr.Dataset]:
    """
    Get a time series dataset from a data *cube*.

    *geometry* may be provided as a (shapely) geometry object, a valid GeoJSON object, a valid WKT string,
    box coordinates (x1, y1, x2, y2), or point coordinates (x, y).

    *start_date* and *end_date* may be provided as a numpy.datetime64 or an ISO datetime string.

    :param cube: The data cube
    :param geometry: Optional geometry
    :param var_names: Optional sequence of names of variables to be included.
    :param start_date: Optional start date.
    :param end_date: Optional end date.
    :param use_groupby: Internal setting.
    :param include_count: Whether to include counts. Ignored if geometry is a point.
    :param include_stdev: Whether to include standard-deviation. Ignored if geometry is a point.
    :return: A new dataset with time-series for each variable.
    """

    if not cube_asserted:
        assert_cube(cube)

    geometry = convert_geometry(geometry)

    dataset = select_vars(cube, var_names)
    if len(dataset.data_vars) == 0:
        return None

    if start_date is not None or end_date is not None:
        # noinspection PyTypeChecker
        dataset = dataset.sel(time=slice(start_date, end_date))

    if isinstance(geometry, shapely.geometry.Point):
        bounds = get_dataset_geometry(dataset)
        if not bounds.contains(geometry):
            return None
        dataset = dataset.sel(lon=geometry.x, lat=geometry.y, method='Nearest')
        return dataset.assign_attrs(max_number_of_observations=1)

    if geometry is not None:
        dataset = where_geometry(dataset, geometry, mask_var_name='__mask__')
        if dataset is None:
            return None
        mask = dataset['__mask__']
        max_number_of_observations = np.count_nonzero(mask)
        dataset = dataset.drop('__mask__')
    else:
        max_number_of_observations = dataset.lat.size * dataset.lon.size

    ds_count = None
    ds_stdev = None
    if use_groupby:
        time_group = dataset.groupby('time')
        ds_mean = time_group.mean(skipna=True)
        if include_count:
            ds_count = time_group.count()
        if include_stdev:
            ds_stdev = time_group.std(skipna=True)
    else:
        ds_mean = dataset.mean(dim=('lat', 'lon'), skipna=True)
        if include_count:
            ds_count = dataset.count(dim=('lat', 'lon'))
        if include_stdev:
            ds_stdev = dataset.std(dim=('lat', 'lon'), skipna=True)

    if ds_count is not None:
        ds_count = ds_count.rename(name_dict=dict({v: f"{v}_count" for v in ds_count.data_vars}))

    if ds_stdev is not None:
        ds_stdev = ds_stdev.rename(name_dict=dict({v: f"{v}_stdev" for v in ds_stdev.data_vars}))

    if ds_count is not None and ds_stdev is not None:
        ts_dataset = xr.merge([ds_mean, ds_stdev, ds_count])
    elif ds_count is not None:
        ts_dataset = xr.merge([ds_mean, ds_count])
    elif ds_stdev is not None:
        ts_dataset = xr.merge([ds_mean, ds_stdev])
    else:
        ts_dataset = ds_mean

    ts_dataset = ts_dataset.assign_attrs(max_number_of_observations=max_number_of_observations)

    return ts_dataset
