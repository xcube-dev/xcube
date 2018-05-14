import warnings
from typing import Tuple, List

import gdal
import numpy as np
import pandas as pd
import xarray as xr

from .constants import CRS_WKT_EPSG_4326, EARTH_GEO_COORD_RANGE
from .types import CoordRange

# TODO: add src and dst CRS
# TODO: add callback: Callable[[Optional[Any, str]], None] = None, callback_data: Any = None
# TODO: support waveband_var_names so that we can combine spectra as vectors

REF_DATETIME_STR = '1970-01-01 00:00:00'
REF_DATETIME = pd.to_datetime(REF_DATETIME_STR)
DATETIME_UNITS = f'days since {REF_DATETIME_STR}'
DATETIME_CALENDAR = 'gregorian'
SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = 1000 * 1000 * SECONDS_PER_DAY

def reproject_to_wgs84(dataset: xr.Dataset,
                       dst_size: Tuple[int, int],
                       dst_region: CoordRange = None,
                       src_crs_wks: str = None,
                       valid_region: CoordRange = None,
                       gcp_i_step: int = 10,
                       gcp_j_step: int = None,
                       tp_gcp_i_step: int = 4,
                       tp_gcp_j_step: int = None,
                       include_non_spatial_vars: bool = False) -> xr.Dataset:
    """
    Reprojection of xarray datasets with 2D geo-coding, e.g. with variables lon(y,x), lat(y, x) to
    EPSG:4326 (WGS-84) coordinate reference system.

    :param dataset:
    :param dst_size:
    :param dst_region:
    :param src_crs_wks:
    :param valid_region:
    :param gcp_i_step:
    :param gcp_j_step:
    :param tp_gcp_i_step:
    :param tp_gcp_j_step:
    :param include_non_spatial_vars:
    :return: the reprojected dataset
    """
    # TODO: Turn following into options. Current default values refer to BEAM/SNAP profile.
    x_name = 'lon'
    y_name = 'lat'
    tp_x_name = 'TP_longitude'
    tp_y_name = 'TP_latitude'
    time_min_name = 'start_date'
    time_max_name = 'stop_date'
    # Example: "14-APR-2017 10:27:50.183264"
    time_format = None  # '%Y-%m-%d %H:%M%:%S'

    # Set defaults
    src_crs_wks = src_crs_wks or CRS_WKT_EPSG_4326
    valid_region = valid_region or EARTH_GEO_COORD_RANGE
    gcp_j_step = gcp_i_step or gcp_j_step
    tp_gcp_j_step = tp_gcp_i_step or tp_gcp_j_step

    dst_width, dst_height = dst_size

    # TODO: raise ValueError instead of using assert
    assert dataset is not None
    assert dst_width > 1
    assert dst_height > 1
    assert gcp_i_step > 0
    assert gcp_j_step > 0

    assert x_name in dataset
    assert y_name in dataset

    x_var = dataset[x_name]
    assert len(x_var.dims) == 2
    assert x_var.shape[-1] >= 2
    assert x_var.shape[-2] >= 2
    y_var = dataset[y_name]
    assert y_var.shape == x_var.shape
    assert y_var.dims == x_var.dims

    src_width = x_var.shape[-1]
    src_height = x_var.shape[-2]

    dst_region = _ensure_valid_region(dst_region, valid_region, x_var, y_var)
    x1, y1, x2, y2 = dst_region

    dst_res = max((x2 - x1) / dst_width, (y2 - y1) / dst_height)
    assert dst_res > 0

    dst_geo_transform = (x1, dst_res, 0.0,
                         y2, 0.0, -dst_res)

    # Extract GCPs from full-res lon/lat 2D variables
    gcps = _get_gcps(x_var, y_var, gcp_i_step, gcp_j_step)

    if tp_x_name in dataset and tp_y_name in dataset:
        tp_x_var = dataset[tp_x_name]
        tp_y_var = dataset[tp_y_name]
        assert len(tp_x_var.shape) == 2
        assert tp_x_var.shape == tp_y_var.shape
        tp_width = tp_x_var.shape[-1]
        tp_height = tp_x_var.shape[-2]
        assert tp_gcp_i_step > 0
        assert tp_gcp_j_step > 0
        tp_gcps = _get_gcps(tp_x_var, tp_y_var, tp_gcp_i_step, tp_gcp_j_step)
    else:
        tp_gcps = None
        tp_width = None
        tp_height = None
        tp_x_var = None

    if 'time' not in dataset:
        t1 = dataset.attrs.get(time_min_name)
        t2 = dataset.attrs.get(time_max_name)
        if t1 is not None:
            t1 = _get_time_in_days_since_1970(t1, format=time_format)
        if t2 is not None:
            t2 = _get_time_in_days_since_1970(t2, format=time_format)
    else:
        t1 = None
        t2 = None

    mem_driver = gdal.GetDriverByName("MEM")

    dst_x2 = x1 + dst_res * dst_width
    dst_y1 = y2 - dst_res * dst_height

    dst_dataset = xr.Dataset()
    dst_dataset.coords['lon'] = (['lon', ],
                                 np.linspace(x1 + dst_res / 2, dst_x2 - dst_res / 2, dst_width),
                                 dict(bounds='lon_bnds', long_name='longitude', units='degrees_east'))
    dst_dataset.coords['lat'] = (['lat', ],
                                 np.linspace(y2 - dst_res / 2, dst_y1 + dst_res / 2, dst_height),
                                 dict(bounds='lat_bnds', long_name='latitude', units='degrees_north'))
    dst_dataset.coords['lon_bnds'] = (['lon', 'bnds'],
                                      # TODO: find more elegant numpy expr for the following
                                      list(zip(np.linspace(x1, dst_x2 - dst_res, dst_width),
                                               np.linspace(x1 + dst_res, dst_x2, dst_width))),
                                      dict(units='degrees_east'))
    dst_dataset.coords['lat_bnds'] = (['lat', 'bnds'],
                                      # TODO: find more elegant numpy expr for the following
                                      list(zip(np.linspace(y2, dst_y1 + dst_res, dst_height),
                                               np.linspace(y2 - dst_res, dst_y1, dst_height))),
                                      dict(units='degrees_north'))
    dst_dataset.attrs = dataset.attrs
    dst_dataset.attrs['Conventions'] = 'CF-1.6'

    for var_name in dataset.variables:
        src_var = dataset[var_name]

        if var_name == x_name or var_name == y_name:
            # Don't store lat and lon 2D vars in destination, this will let xarray raise
            # TODO: instead add lat and lon with new names, so we can validate geo-location
            continue

        is_tp_var = False
        if src_var.dims == x_var.dims:
            # TODO: collect variables of same type and size and set band_count accordingly to speed up reprojection
            band_count = 1
            # TODO: select the data_type based on src_var.dtype
            data_type = gdal.GDT_Float64
            src_var_dataset = mem_driver.Create('src_' + var_name, src_width, src_height, band_count, data_type, [])
            src_var_dataset.SetGCPs(gcps, src_crs_wks)
        elif tp_x_var is not None and src_var.dims == tp_x_var.dims:
            if var_name == tp_y_name or var_name == tp_x_name:
                # Don't store TP lat and TP lon 2D vars in destination
                # TODO: instead add TP lat and lon with new names, so we can validate geo-location
                continue
            is_tp_var = True
            # TODO: collect variables of same type and size and set band_count accordingly to speed up reprojection
            band_count = 1
            # TODO: select the data_type based on src_var.dtype
            data_type = gdal.GDT_Float64
            src_var_dataset = mem_driver.Create('src_' + var_name, tp_width, tp_height, band_count, data_type, [])
            src_var_dataset.SetGCPs(tp_gcps, src_crs_wks)
        elif include_non_spatial_vars:
            # Store any variable as-is, that does not have the lat/lon 2D dims, then continue
            dst_dataset[var_name] = src_var
            continue
        else:
            continue

        # We use GDT_Float64 to introduce NaN as no-data-value
        dst_data_type = gdal.GDT_Float64
        dst_var_dataset = mem_driver.Create('dst_' + var_name, dst_width, dst_height, band_count, dst_data_type, [])
        dst_var_dataset.SetProjection(CRS_WKT_EPSG_4326)
        dst_var_dataset.SetGeoTransform(dst_geo_transform)

        for band_index in range(1, band_count + 1):
            src_var_dataset.GetRasterBand(band_index).SetNoDataValue(float('nan'))
            src_var_dataset.GetRasterBand(band_index).WriteArray(src_var.values)
            dst_var_dataset.GetRasterBand(band_index).SetNoDataValue(float('nan'))

        # TODO: configure resampling individually for each variable, make this config a parameter
        resampling = gdal.GRA_Bilinear if is_tp_var else gdal.GRA_NearestNeighbour
        warp_mem_limit = 0
        error_threshold = 0
        # See http://www.gdal.org/structGDALWarpOptions.html
        options = ['INIT_DEST=NO_DATA']
        gdal.ReprojectImage(src_var_dataset,
                            dst_var_dataset,
                            None,
                            None,
                            resampling,
                            warp_mem_limit,
                            error_threshold,
                            None,  # callback,
                            None,  # callback_data,
                            options)  # options

        dst_values = dst_var_dataset.GetRasterBand(1).ReadAsArray()
        # print(var_name, dst_values.shape, np.nanmin(dst_values), np.nanmax(dst_values))

        # TODO: set CF-1.6 attributes correctly
        dst_var = xr.DataArray(dst_values, dims=['lat', 'lon'], name=var_name, attrs=src_var.attrs)
        dst_var.encoding = src_var.encoding
        if np.issubdtype(dst_var.dtype, np.floating) \
                and np.issubdtype(src_var.encoding.get('dtype'), np.integer) \
                and src_var.encoding.get('_FillValue') is None:
            warnings.warn(f'variable {dst_var.name!r}: setting _FillValue=0 to replace any NaNs')
            dst_var.encoding['_FillValue'] = 0
        dst_var.encoding['chunksizes'] = (1, dst_var.shape[-2], dst_var.shape[-1])
        dst_var.encoding['zlib'] = True
        dst_var.encoding['complevel'] = 4
        dst_dataset[var_name] = dst_var

    if t1 or t2:
        dst_dataset = dst_dataset.expand_dims('time')
        if t1 and t2:
            t_center = t1 + 0.5 * (t2 - t1)
        else:
            t_center = t1 or t2
        dst_dataset = dst_dataset.assign_coords(time=(['time'], [t_center]))
        time_var = dst_dataset.coords['time']
        time_var.attrs['long_name'] = 'time'
        time_var.attrs['standard_name'] = 'time'
        time_var.attrs['units'] = DATETIME_UNITS
        time_var.attrs['calendar'] = DATETIME_CALENDAR
        time_var.encoding['units'] = DATETIME_UNITS
        time_var.encoding['calendar'] = DATETIME_CALENDAR
        if t1 and t2:
            time_var.attrs['bounds'] = 'time_bnds'
            dst_dataset = dst_dataset.assign_coords(time_bnds=(['time', 'bnds'], [[t1, t2]]))
            time_bnds_var = dst_dataset.coords['time_bnds']
            time_bnds_var.attrs['long_name'] = 'time'
            time_bnds_var.attrs['standard_name'] = 'time'
            time_bnds_var.attrs['units'] = DATETIME_UNITS
            time_bnds_var.attrs['calendar'] = DATETIME_CALENDAR
            time_bnds_var.encoding['units'] = DATETIME_UNITS
            time_bnds_var.encoding['calendar'] = DATETIME_CALENDAR

    return dst_dataset


def _ensure_valid_region(region: CoordRange,
                         valid_region: CoordRange,
                         x_var: xr.DataArray,
                         y_var: xr.DataArray,
                         extra: float = 0.01):
    if region:
        # Extract region
        assert len(region) == 4
        x1, y1, x2, y2 = region
    else:
        # Determine region from full-res lon/lat 2D variables
        x1, x2 = x_var.min(), x_var.max()
        y1, y2 = y_var.min(), y_var.max()
        extra_space = extra * max(x2 - x1, y2 - y1)
        # Add extra space in units of the source coordinates
        x1, x2 = x1 - extra_space, x2 + extra_space
        y1, y2 = y1 - extra_space, y2 + extra_space
    x_min, y_min, x_max, y_max = valid_region
    x1, x2 = max(x_min, x1), min(x_max, x2)
    y1, y2 = max(y_min, y1), min(y_max, y2)
    assert x1 < x2 and y1 < y2
    return x1, y1, x2, y2


def _get_gcps(x_var: xr.DataArray,
              y_var: xr.DataArray,
              i_step: int,
              j_step: int) -> List[gdal.GCP]:
    x_values = x_var.values
    y_values = y_var.values
    i_size = x_var.shape[-1]
    j_size = x_var.shape[-2]
    gcps = []
    gcp_id = 0
    i_count = (i_size + i_step - 1) // i_step
    j_count = (j_size + j_step - 1) // j_step
    for j in np.linspace(0, j_size - 1, j_count, dtype=np.int32):
        for i in np.linspace(0, i_size - 1, i_count, dtype=np.int32):
            x, y = float(x_values[j, i]), float(y_values[j, i])
            gcps.append(gdal.GCP(x, y, 0.0, i + 0.5, j + 0.5, '%s,%s' % (i, j), str(gcp_id)))
            gcp_id += 1
    return gcps


def _get_time_in_days_since_1970(t: str, format=None) -> float:
    datetime = pd.to_datetime(t, format=format, infer_datetime_format=True)
    timedelta = datetime - REF_DATETIME
    return timedelta.days + timedelta.seconds / SECONDS_PER_DAY + timedelta.microseconds / MICROSECONDS_PER_DAY
