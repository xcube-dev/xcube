from typing import Collection, Optional, Tuple
from typing import Union

import cftime
import pandas as pd
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.util.assertions import assert_given

# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

Bbox = Tuple[float, float, float, float]
TimeRange = Union[Tuple[Optional[str], Optional[str]],
                  Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]


def select_subset(dataset: xr.Dataset,
                  *,
                  var_names: Collection[str] = None,
                  bbox: Bbox = None,
                  time_range: TimeRange = None):
    """
    Create a subset from *dataset* given *var_names*,
    *bbox*, *time_range*.

    This is a high-level convenience function that may invoke

    * :func:select_variables_subset
    * :func:select_spatial_subset
    * :func:select_temporal_subset

    :param dataset: The dataset.
    :param var_names: Optional variable names.
    :param bbox: Optional bounding box in the dataset's
        CRS coordinate units.
    :param time_range: Optional time range
    :return: a subset of *dataset*, or unchanged *dataset*
        if no keyword-arguments are used.
    """
    if var_names is not None:
        dataset = select_variables_subset(dataset, var_names=var_names)
    if bbox is not None:
        dataset = select_spatial_subset(dataset, xy_bbox=bbox)
    if time_range is not None:
        dataset = select_temporal_subset(dataset, time_range=time_range)
    return dataset


def select_variables_subset(dataset: xr.Dataset,
                            var_names: Collection[str] = None) -> xr.Dataset:
    """
    Select data variable from given *dataset* and create new dataset.

    :param dataset: The dataset from which to select variables.
    :param var_names: The names of data variables to select.
    :return: A new dataset. It is empty, if *var_names* is empty.
        It is *dataset*, if *var_names* is None.
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
                          grid_mapping: GridMapping = None,
                          geo_coding: GridMapping = None,
                          xy_names: Tuple[str, str] = None) \
        -> Optional[xr.Dataset]:
    """
    Select a spatial subset of *dataset* for the
    bounding box *ij_bbox* or *xy_bbox*.

    *ij_bbox* or *xy_bbox* must not be given both.

    :param xy_bbox: Bounding box in coordinates of the dataset's CRS.
    :param xy_border: Extra border added to *xy_bbox*.
    :param dataset: Source dataset.
    :param ij_bbox: Bounding box (i_min, i_min, j_max, j_max)
        in pixel coordinates.
    :param ij_border: Extra border added to *ij_bbox*
        in number of pixels
    :param xy_bbox: The bounding box in x,y coordinates.
    :param xy_border: Border in units of the x,y coordinates.
    :param grid_mapping: Optional dataset grid mapping.
    :param geo_coding: Deprecated. Use *grid_mapping* instead.
    :param xy_names: Optional tuple of the x- and y-coordinate
        variables in *dataset*. Ignored if *geo_coding* is given.
    :return: Spatial dataset subset
    """

    if ij_bbox is None and xy_bbox is None:
        raise ValueError('One of ij_bbox and xy_bbox must be given')
    if ij_bbox and xy_bbox:
        raise ValueError('Only one of ij_bbox and xy_bbox can be given')

    grid_mapping = geo_coding or grid_mapping
    if grid_mapping is None:
        grid_mapping = GridMapping.from_dataset(dataset,
                                                xy_var_names=xy_names)
    x_name, y_name = grid_mapping.xy_var_names
    x = dataset[x_name]
    y = dataset[y_name]

    if x.ndim == 1 and y.ndim == 1:
        # Hotfix für #981 and #985
        if xy_bbox:
            if y.values[0] < y.values[-1]:
                ds = dataset.sel(**{
                    x_name: slice(xy_bbox[0] - xy_border,
                                  xy_bbox[2] + xy_border),
                    y_name: slice(xy_bbox[1] - xy_border,
                                  xy_bbox[3] + xy_border)
                })
            else:
                ds = dataset.sel(**{
                    x_name: slice(xy_bbox[0] - xy_border,
                                  xy_bbox[2] + xy_border),
                    y_name: slice(xy_bbox[3] + xy_border,
                                  xy_bbox[1] - xy_border)
                })
            return ds
        else:
            return dataset.isel(**{
                x_name: slice(ij_bbox[0] - ij_border,
                              ij_bbox[2] + ij_border),
                y_name: slice(ij_bbox[1] - ij_border,
                              ij_bbox[3] + ij_border)
            })
    else:
        if xy_bbox:
            ij_bbox = grid_mapping.ij_bbox_from_xy_bbox(xy_bbox,
                                                        ij_border=ij_border,
                                                        xy_border=xy_border)
            if ij_bbox[0] == -1:
                return None
        width, height = grid_mapping.size
        i_min, j_min, i_max, j_max = ij_bbox
        if i_min > 0 or j_min > 0 or i_max < width - 1 or j_max < height - 1:
            x_dim, y_dim = grid_mapping.xy_dim_names
            i_slice = slice(i_min, i_max + 1)
            j_slice = slice(j_min, j_max + 1)
            return dataset.isel({x_dim: i_slice, y_dim: j_slice})
        return dataset


def select_temporal_subset(dataset: xr.Dataset,
                           time_range: TimeRange,
                           time_name: str = 'time') -> xr.Dataset:
    """
    Select a temporal subset from *dataset* given *time_range*.

    :param dataset: The dataset. Must include time
    :param time_range: Time range given as two time stamps
        (start, end) that may be (ISO) strings or datetime objects.
    :param time_name: optional name of the time coordinate variable.
        Defaults to "time".
    :return:
    """
    assert_given(time_range, 'time_range')
    time_name = time_name or 'time'
    if time_name not in dataset:
        raise ValueError(f'cannot compute temporal subset: variable'
                         f' "{time_name}" not found in dataset')
    time_1, time_2 = time_range
    time_1 = pd.to_datetime(time_1) if time_1 is not None else None
    time_2 = pd.to_datetime(time_2) if time_2 is not None else None
    if time_1 is None and time_2 is None:
        return dataset
    if time_2 is not None:
        delta = time_2 - time_2.floor('1D')
        if delta == pd.Timedelta('0 days 00:00:00'):
            time_2 += pd.Timedelta('1D')
    try:
        return dataset.sel({time_name or 'time': slice(time_1, time_2)})
    except TypeError:
        calendar = dataset.time.encoding.get('calendar')
        time_1 = cftime.datetime(time_1.year, time_1.month, time_1.day,
                                 calendar=calendar)
        time_2 = cftime.datetime(time_2.year, time_2.month, time_2.day,
                                 calendar=calendar)
        return dataset.sel({time_name or 'time': slice(time_1, time_2)})
