# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

import warnings
from datetime import datetime
from typing import Optional, Sequence, Union, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from jdcal import jd2gcal


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize the geo- and time-coding upon opening the given dataset w.r.t.
    to a common (CF-compatible) convention used within Cate. This will maximize the compatibility of
    a dataset for usage with Cate's operations.

    That is,
    * variables named "latitude" will be renamed to "lat";
    * variables named "longitude" or "long" will be renamed to "lon";

    Then, for equi-rectangular grids,
    * Remove 2D "lat" and "lon" variables;
    * Two new 1D coordinate variables "lat" and "lon" will be generated from original 2D forms.

    Then, if no coordinate variable time is present but the CF attributes "time_coverage_start" and optionally
    "time_coverage_end" are given, a scalar time dimension and coordinate variable will be generated.

    Finally, it will be ensured that a "time" coordinate variable will be of type *datetime*.

    :param ds: The dataset to normalize.
    :return: The normalized dataset, or the original dataset, if it is already "normal".
    """
    ds = normalize_coord_vars(ds)
    ds = _normalize_lat_lon(ds)
    ds = _normalize_lat_lon_2d(ds)
    ds = _normalize_dim_order(ds)
    ds = _normalize_lon_360(ds)
    ds = _normalize_inverted_lat(ds)
    ds = normalize_missing_time(ds)
    ds = _normalize_jd2datetime(ds)
    return ds


def _normalize_inverted_lat(ds: xr.Dataset) -> xr.Dataset:
    """
    In case the latitude decreases, invert it
    :param ds: some xarray dataset
    :return: a normalized xarray dataset
    """
    try:
        if _lat_inverted(ds.lat):
            ds = ds.sel(lat=slice(None, None, -1))
    except AttributeError:
        # The dataset doesn't have 'lat', probably not geospatial
        pass
    except ValueError:
        # The dataset still has an ND 'lat' array
        pass
    return ds


def _normalize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """
    Rename variables named 'longitude' or 'long' to 'lon', and 'latitude' to 'lon'.
    :param ds: some xarray dataset
    :return: a normalized xarray dataset, or the original one
    """
    lat_name = get_lat_dim_name_impl(ds)
    lon_name = get_lon_dim_name_impl(ds)

    name_dict = dict()
    if lat_name and 'lat' not in ds:
        name_dict[lat_name] = 'lat'

    if lon_name and 'lon' not in ds:
        name_dict[lon_name] = 'lon'

    if name_dict:
        ds = ds.rename(name_dict)

    return ds


def _normalize_lat_lon_2d(ds: xr.Dataset) -> xr.Dataset:
    """
    Detect 2D 'lat', 'lon' variables that span a equi-rectangular grid. Then:
    Drop original 'lat', 'lon' variables
    Rename original dimensions names of 'lat', 'lon' variables, usually ('y', 'x'), to ('lat', 'lon').
    Insert new 1D 'lat', 'lon' coordinate variables with dimensions 'lat' and 'lon', respectively.
    :param ds: some xarray dataset
    :return: a normalized xarray dataset, or the original one
    """
    if not ('lat' in ds and 'lon' in ds):
        return ds

    lat_var = ds['lat']
    lon_var = ds['lon']

    lat_dims = lat_var.dims
    lon_dims = lon_var.dims
    if lat_dims != lon_dims:
        return ds

    spatial_dims = lon_dims
    if len(spatial_dims) != 2:
        return ds

    x_dim_name = spatial_dims[-1]
    y_dim_name = spatial_dims[-2]

    lat_data_1 = lat_var[:, 0]
    lat_data_2 = lat_var[:, -1]
    lon_data_1 = lon_var[0, :]
    lon_data_2 = lon_var[-1, :]

    equal_lat = np.allclose(lat_data_1, lat_data_2, equal_nan=True)
    equal_lon = np.allclose(lon_data_1, lon_data_2, equal_nan=True)

    # Drop lat lon in any case. If note qual_lat and equal_lon subset_spatial_impl will subsequently
    # fail with a ValidationError

    ds = ds.drop_vars(['lon', 'lat'])

    if not (equal_lat and equal_lon):
        return ds

    ds = ds.rename({
        x_dim_name: 'lon',
        y_dim_name: 'lat',
    })

    ds = ds.assign_coords(lon=np.array(lon_data_1), lat=np.array(lat_data_1))

    return ds


def _normalize_lon_360(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix the longitude of the given dataset ``ds`` so that it ranges from -180 to +180 degrees.

    :param ds: The dataset whose longitudes may be given in the range 0 to 360.
    :return: The fixed dataset or the original dataset.
    """

    if 'lon' not in ds.coords:
        return ds

    lon_var = ds.coords['lon']

    if len(lon_var.shape) != 1:
        return ds

    lon_size = lon_var.shape[0]
    if lon_size < 2:
        return ds

    lon_size_05 = lon_size // 2
    lon_values = lon_var.values
    if not np.any(lon_values[lon_size_05:] > 180.):
        return ds

    delta_lon = lon_values[1] - lon_values[0]

    var_names = [var_name for var_name in ds.data_vars]

    ds = ds.assign_coords(lon=xr.DataArray(np.linspace(-180. + 0.5 * delta_lon,
                                                       +180. - 0.5 * delta_lon,
                                                       lon_size),
                                           dims=ds['lon'].dims,
                                           attrs=dict(long_name='longitude',
                                                      standard_name='longitude',
                                                      units='degrees east')))

    ds = adjust_spatial_attrs_impl(ds, True)

    new_vars = dict()
    for var_name in var_names:
        var = ds[var_name]
        if len(var.dims) >= 1 and var.dims[-1] == 'lon':
            values = np.copy(var.values)
            temp = np.copy(values[..., : lon_size_05])
            values[..., : lon_size_05] = values[..., lon_size_05:]
            values[..., lon_size_05:] = temp
            new_var = xr.DataArray(values, dims=var.dims, attrs=var.attrs)
            new_var.encoding.update(var.encoding)
            new_vars[var_name] = new_var

    return ds.assign(**new_vars)


def _normalize_jd2datetime(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert the time dimension of the given dataset from Julian date to
    datetime.

    :param ds: Dataset on which to run conversion
    """

    try:
        time = ds.time
    except AttributeError:
        return ds

    try:
        units = time.units
    except AttributeError:
        units = None

    try:
        long_name = time.long_name
    except AttributeError:
        long_name = None

    if units:
        units = units.lower().strip()

    if long_name:
        units = long_name.lower().strip()

    units = units or long_name

    if not units or units != 'time in julian days':
        return ds

    ds = ds.copy()
    # Decode JD time
    # noinspection PyTypeChecker
    tuples = [jd2gcal(x, 0) for x in ds.time.values]
    # Replace JD time with datetime
    ds['time'] = [datetime(x[0], x[1], x[2]) for x in tuples]
    # Adjust attributes
    ds.time.attrs['long_name'] = 'time'
    ds.time.attrs['calendar'] = 'standard'

    return ds


def normalize_coord_vars(ds: xr.Dataset) -> xr.Dataset:
    """
    Turn potential coordinate variables from data variables into coordinate variables.

    Any data variable is considered a coordinate variable

    * whose name is its only dimension name;
    * whose number of dimensions is two and where the first dimension name is also a variable namd and
      whose last dimension is named "bnds".

    :param ds: The dataset
    :return: The same dataset or a shallow copy with potential coordinate
             variables turned into coordinate variables.
    """

    if 'bnds' not in ds.dims:
        return ds

    coord_var_names = set()
    for data_var_name in ds.data_vars:
        data_var = ds.data_vars[data_var_name]
        if is_coord_var(ds, data_var):
            coord_var_names.add(data_var_name)

    if not coord_var_names:
        return ds

    old_ds = ds
    ds = old_ds.drop_vars(coord_var_names)
    ds = ds.assign_coords(**{bounds_var_name: old_ds[bounds_var_name] for bounds_var_name in coord_var_names})

    return ds


def normalize_missing_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Add a time coordinate variable and their associated bounds coordinate variables
    if temporal CF attributes ``time_coverage_start`` and ``time_coverage_end``
    are given but the time dimension is missing.

    The new time coordinate variable will be named ``time`` with dimension ['time'] and shape [1].
    The time bounds coordinates variable will be named ``time_bnds`` with dimensions ['time', 'bnds'] and shape [1,2].
    Both are of data type ``datetime64``.

    :param ds: Dataset to adjust
    :return: Adjusted dataset
    """
    time_coverage_start = ds.attrs.get('time_coverage_start')
    if time_coverage_start is not None:
        # noinspection PyBroadException
        try:
            time_coverage_start = pd.to_datetime(time_coverage_start)
        except BaseException:
            pass

    time_coverage_end = ds.attrs.get('time_coverage_end')
    if time_coverage_end is not None:
        # noinspection PyBroadException
        try:
            time_coverage_end = pd.to_datetime(time_coverage_end)
        except BaseException:
            pass

    if not time_coverage_start and not time_coverage_end:
        # Can't do anything
        return ds

    if 'time' in ds:
        time = ds.time
        if not time.dims:
            ds = ds.drop_vars('time')
        elif len(time.dims) == 1:
            time_dim_name = time.dims[0]
            is_time_used_as_dim = any([(time_dim_name in ds[var_name].dims) for var_name in ds.data_vars])
            if is_time_used_as_dim:
                # It seems we already have valid time coordinates
                return ds
            time_bnds_var_name = time.attrs.get('bounds')
            if time_bnds_var_name in ds:
                ds = ds.drop_vars(time_bnds_var_name)
            ds = ds.drop_vars('time')
            ds = ds.drop_vars([var_name for var_name in ds.coords if time_dim_name in ds.coords[var_name].dims])

    if time_coverage_start or time_coverage_end:
        # noinspection PyBroadException
        try:
            ds = ds.expand_dims('time')
        except BaseException as e:
            warnings.warn(f'failed to add time dimension: {e}')

        if time_coverage_start and time_coverage_end:
            time_value = time_coverage_start + 0.5 * (time_coverage_end - time_coverage_start)
        else:
            time_value = time_coverage_start or time_coverage_end

        new_coord_vars = dict(time=xr.DataArray([time_value], dims=['time']))

        if time_coverage_start and time_coverage_end:
            has_time_bnds = 'time_bnds' in ds.coords or 'time_bnds' in ds
            if not has_time_bnds:
                new_coord_vars.update(time_bnds=xr.DataArray([[time_coverage_start, time_coverage_end]],
                                                             dims=['time', 'bnds']))

        ds = ds.assign_coords(**new_coord_vars)

        ds.coords['time'].attrs['long_name'] = 'time'
        ds.coords['time'].attrs['standard_name'] = 'time'
        ds.coords['time'].encoding['units'] = 'days since 1970-01-01'
        if 'time_bnds' in ds.coords:
            ds.coords['time'].attrs['bounds'] = 'time_bnds'
            ds.coords['time_bnds'].attrs['long_name'] = 'time'
            ds.coords['time_bnds'].attrs['standard_name'] = 'time'
            ds.coords['time_bnds'].encoding['units'] = 'days since 1970-01-01'

    return ds


def adjust_spatial_attrs_impl(ds: xr.Dataset, allow_point: bool) -> xr.Dataset:
    """
    Adjust the global spatial attributes of the dataset by doing some
    introspection of the dataset and adjusting the appropriate attributes
    accordingly.

    In case the determined attributes do not exist in the dataset, these will
    be added.

    For more information on suggested global attributes see
    `Attribute Convention for Data Discovery
    <http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery>`_

    :param ds: Dataset to adjust
    :param allow_point: Whether to accept single point cells
    :return: Adjusted dataset
    """

    copied = False

    for dim in ('lon', 'lat'):
        geo_spatial_attrs = _get_geo_spatial_cf_attrs_from_var(ds, dim, allow_point=allow_point)
        if geo_spatial_attrs:
            # Copy any new attributes into the shallow Dataset copy
            for key in geo_spatial_attrs:
                if geo_spatial_attrs[key] is not None:
                    if not copied:
                        ds = ds.copy()
                        copied = True
                    ds.attrs[key] = geo_spatial_attrs[key]

    lon_min = ds.attrs.get('geospatial_lon_min')
    lat_min = ds.attrs.get('geospatial_lat_min')
    lon_max = ds.attrs.get('geospatial_lon_max')
    lat_max = ds.attrs.get('geospatial_lat_max')

    if lon_min is not None and lat_min is not None and lon_max is not None and lat_max is not None:

        if not copied:
            ds = ds.copy()

        ds.attrs['geospatial_bounds'] = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'. \
            format(lon_min, lat_min, lon_min, lat_max, lon_max, lat_max, lon_max, lat_min, lon_min, lat_min)

        # Determination of the following attributes from introspection in a general
        # way is ambiguous, hence it is safer to drop them than to risk preserving
        # out of date attributes.
        drop = ['geospatial_bounds_crs', 'geospatial_bounds_vertical_crs',
                'geospatial_vertical_min', 'geospatial_vertical_max',
                'geospatial_vertical_positive', 'geospatial_vertical_units',
                'geospatial_vertical_resolution']

        for key in drop:
            ds.attrs.pop(key, None)

    return ds


def adjust_temporal_attrs_impl(ds: xr.Dataset) -> xr.Dataset:
    """
    Adjust the global temporal attributes of the dataset by doing some
    introspection of the dataset and adjusting the appropriate attributes
    accordingly.

    In case the determined attributes do not exist in the dataset, these will
    be added.

    For more information on suggested global attributes see
    `Attribute Convention for Data Discovery
    <http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery>`_

    :param ds: Dataset to adjust
    :return: Adjusted dataset
    """

    temporal_attrs = _get_temporal_cf_attrs_from_var(ds)

    if temporal_attrs:
        ds = ds.copy()
        # Align temporal attributes with the ones from the shallow Dataset copy
        for key in temporal_attrs:
            if temporal_attrs[key] is not None:
                ds.attrs[key] = temporal_attrs[key]
            else:
                ds.attrs.pop(key, None)

    return ds


def is_coord_var(ds: xr.Dataset, var: xr.DataArray):
    if len(var.dims) == 1 and var.name == var.dims[0]:
        return True
    return is_bounds_var(ds, var)


def is_bounds_var(ds: xr.Dataset, var: xr.DataArray):
    if len(var.dims) == 2 \
            and var.shape[1] == 2 \
            and var.dims[1] == 'bnds':
        coord_name = var.dims[0]
        return coord_name in ds
    return False


def _get_temporal_cf_attrs_from_var(ds: xr.Dataset, var_name: str = 'time') -> Optional[dict]:
    """
    Get temporal boundaries, resolution and duration of the given dataset. If
    the 'bounds' are explicitly defined, these will be used for calculation,
    otherwise it will rest on information gathered from the 'time' dimension
    itself.

    :param ds: Dataset
    :param var_name: The variable/dimension name.
    :return: A dictionary {'attr_name': attr_value}
    """

    if var_name not in ds:
        return None

    var = ds[var_name]

    if 'bounds' in var.attrs:
        # According to CF Conventions the corresponding 'bounds' coordinate variable name
        # should be in the attributes of the coordinate variable
        bnds_name = var.attrs['bounds']
    else:
        # If 'bounds' attribute is missing, the bounds coordinate variable may be named "<dim>_bnds"
        bnds_name = f'{var_name}_bnds'

    dim_min = dim_max = None
    dim_var = None

    if bnds_name in ds:
        bnds_var = ds[bnds_name]
        if len(bnds_var.shape) == 2 and bnds_var.shape[0] > 0 and bnds_var.shape[1] == 2:
            dim_var = bnds_var
            if bnds_var.shape[0] > 1:
                dim_min = bnds_var.values[0][0]
                dim_max = bnds_var.values[-1][1]
            else:
                dim_min = bnds_var.values[0][0]
                dim_max = bnds_var.values[0][1]

    if dim_var is None:
        if len(var.shape) == 1 and var.shape[0] > 0:
            dim_var = var
            if var.shape[0] > 1:
                dim_min = var.values[0]
                dim_max = var.values[-1]
            else:
                dim_min = var.values[0]
                dim_max = var.values[0]

    # Make sure dim_min and dim_max are valid and are instances of np.datetime64
    # See https://github.com/CCI-Tools/cate/issues/643
    if dim_var is None \
            or not np.issubdtype(dim_var.dtype, np.datetime64):
        # Cannot determine temporal extent for dimension var_name
        return None

    if dim_min != dim_max:
        duration = _get_duration(dim_min, dim_max)
    else:
        duration = None

    if dim_min < dim_max and len(var) >= 2:
        resolution = _get_temporal_res(var.values)
    else:
        resolution = None

    return dict(time_coverage_start=str(dim_min),
                time_coverage_end=str(dim_max),
                time_coverage_duration=duration,
                time_coverage_resolution=resolution)


def _get_geo_spatial_cf_attrs_from_var(ds: xr.Dataset, var_name: str, allow_point: bool = False) -> Optional[dict]:
    """
    Get spatial boundaries, resolution and units of the given dimension of the given
    dataset. If the 'bounds' are explicitly defined, these will be used for
    boundary calculation, otherwise it will rest purely on information gathered
    from 'dim' itself.

    :param ds: The dataset
    :param var_name: The variable/dimension name.
    :param allow_point: True, if it is ok to have no actual spatial extent.
    :return: A dictionary {'attr_name': attr_value}
    """

    if var_name not in ds:
        return None

    var = ds[var_name]

    if 'bounds' in var.attrs:
        # According to CF Conventions the corresponding 'bounds' coordinate variable name
        # should be in the attributes of the coordinate variable
        bnds_name = var.attrs['bounds']
    else:
        # If 'bounds' attribute is missing, the bounds coordinate variable may be named "<dim>_bnds"
        bnds_name = '%s_bnds' % var_name

    dim_var = None

    if bnds_name in ds:
        bnds_var = ds[bnds_name]
        if len(bnds_var.shape) == 2 and bnds_var.shape[0] > 0 and bnds_var.shape[1] == 2:
            dim_var = bnds_var
            dim_res = abs(bnds_var.values[0][1] - bnds_var.values[0][0])
            if bnds_var.shape[0] > 1:
                dim_min = min(bnds_var.values[0][0], bnds_var.values[-1][1])
                dim_max = max(bnds_var.values[0][0], bnds_var.values[-1][1])
            else:
                dim_min = min(bnds_var.values[0][0], bnds_var.values[0][1])
                dim_max = max(bnds_var.values[0][0], bnds_var.values[0][1])

    if dim_var is None:
        if len(var.shape) == 1 and var.shape[0] > 0:
            if var.shape[0] > 1:
                dim_var = var
                dim_res = abs(var.values[1] - var.values[0])
                dim_min = min(var.values[0], var.values[-1]) - 0.5 * dim_res
                dim_max = max(var.values[0], var.values[-1]) + 0.5 * dim_res
            elif len(var.values) == 1 and allow_point:
                dim_var = var
                # Actually a point with no extent
                dim_res = 0.0
                dim_min = var.values[0]
                dim_max = var.values[0]

    if dim_var is None:
        # Cannot determine spatial extent for variable/dimension var_name
        return None

    if 'units' in var.attrs:
        dim_units = var.attrs['units']
    else:
        dim_units = None

    res_name = 'geospatial_{}_resolution'.format(var_name)
    min_name = 'geospatial_{}_min'.format(var_name)
    max_name = 'geospatial_{}_max'.format(var_name)
    units_name = 'geospatial_{}_units'.format(var_name)

    geo_spatial_attrs = dict()
    # noinspection PyUnboundLocalVariable
    geo_spatial_attrs[res_name] = float(dim_res)
    # noinspection PyUnboundLocalVariable
    geo_spatial_attrs[min_name] = float(dim_min)
    # noinspection PyUnboundLocalVariable
    geo_spatial_attrs[max_name] = float(dim_max)
    geo_spatial_attrs[units_name] = dim_units

    return geo_spatial_attrs


def _get_temporal_res(time: np.ndarray) -> str:
    """
    Determine temporal resolution of the given datetimes array.

    See also: `ISO 8601 Durations <https://en.wikipedia.org/wiki/ISO_8601#Durations>`_

    :param time: A numpy array containing np.datetime64 objects
    :return: Temporal resolution formatted as an ISO 8601:2004 duration string
    """
    delta = time[1] - time[0]
    days = delta.astype('timedelta64[D]') / np.timedelta64(1, 'D')

    if (27 < days) and (days < 32):
        return 'P1M'
    else:
        return 'P{}D'.format(int(days))


def _get_duration(tmin: np.datetime64, tmax: np.datetime64) -> str:
    """
    Determine the duration of the given datetimes.

    See also: `ISO 8601 Durations <https://en.wikipedia.org/wiki/ISO_8601#Durations>`_

    :param tmin: Time minimum
    :param tmax: Time maximum
    :return: Temporal resolution formatted as an ISO 8601:2004 duration string
    """
    delta = tmax - tmin
    day = np.timedelta64(1, 'D')
    days = (delta.astype('timedelta64[D]') / day) + 1
    return 'P{}D'.format(int(days))


def get_lon_dim_name_impl(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """
    Get the name of the longitude dimension.
    :param ds: An xarray Dataset
    :return: the name or None
    """
    return _get_dim_name(ds, ['lon', 'longitude', 'long'])


def get_lat_dim_name_impl(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """
    Get the name of the latitude dimension.
    :param ds: An xarray Dataset
    :return: the name or None
    """
    return _get_dim_name(ds, ['lat', 'latitude'])


def _get_dim_name(ds: Union[xr.Dataset, xr.DataArray], possible_names: Sequence[str]) -> Optional[str]:
    for name in possible_names:
        if name in ds.dims:
            return name
    return None


def _lat_inverted(lat: xr.DataArray) -> bool:
    """
    Determine if the latitude is inverted
    """
    if lat.values[0] > lat.values[-1]:
        return True

    return False


def _normalize_dim_order(ds: xr.Dataset) -> xr.Dataset:

    copy_created = False

    for var_name in ds.data_vars:
        var = ds[var_name]
        dim_names = list(var.dims)
        num_dims = len(dim_names)
        if num_dims == 0:
            continue

        must_transpose = False

        if 'time' in dim_names:
            time_index = dim_names.index('time')
            if time_index > 0:
                must_transpose = _swap_pos(dim_names, time_index, 0)

        if num_dims >= 2 and 'lat' in dim_names and 'lon' in dim_names:
            lat_index = dim_names.index('lat')
            if lat_index != num_dims - 2:
                must_transpose = _swap_pos(dim_names, lat_index, -2)
            lon_index = dim_names.index('lon')
            if lon_index != num_dims - 1:
                must_transpose = _swap_pos(dim_names, lon_index, -1)

        if must_transpose:
            if not copy_created:
                ds = ds.copy()
                copy_created = True
            ds[var_name] = var.transpose(*dim_names)

    return ds


def _swap_pos(lst, i1, i2):
    e1, e2 = lst[i1], lst[i2]
    lst[i2], lst[i1] = e1, e2
    return True
