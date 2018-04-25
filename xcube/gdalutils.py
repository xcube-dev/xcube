from typing import Tuple

import gdal
import xarray as xr
import numpy as np

from .constants import CRS_WKT_EPSG_4326

GeoRange = Tuple[float, float, float, float]


def reproject_xarray(dataset: xr.Dataset,
                     dst_width: int,
                     dst_height: int,
                     region: GeoRange = None,
                     gcp_step: int = 10) -> xr.Dataset:
    assert dataset is not None
    assert dst_width > 0
    assert dst_height > 0
    assert gcp_step > 0

    lon_var = dataset.lon
    assert len(lon_var.dims) == 2

    lat_var = dataset.lat
    assert lat_var.shape == lon_var.shape
    assert lat_var.dims == lon_var.dims

    lon_values = lon_var.values
    lat_values = lat_var.values

    src_width = lon_values.shape[-1]
    src_height = lon_values.shape[-2]

    if not region:
        lon_min, lon_max = lon_var.min(), lon_var.max()
        lat_min, lat_max = lat_var.min(), lat_var.max()
        extra = 0.01 * max(lon_max - lon_min, lat_max - lat_min)
        lon_min, lon_max = max(-180., lon_min - extra), min(180., lon_max + extra)
        lat_min, lat_max = max(-90., lat_min - extra), min(90., lat_max + extra)
    else:
        assert len(region) == 4
        lon_min, lat_min, lon_max, lat_max = region

    res = max((lon_max - lon_min) / dst_width, (lat_max - lat_min) / dst_height)
    assert res > 0

    dst_geo_transform = (lon_min, res, 0.0,
                         lat_max, 0.0, -res)

    gcps = []
    for y in range(0, src_height, gcp_step):
        for x in range(0, src_width, gcp_step):
            lon, lat = lon_values[y, x], lat_values[y, x]
            gcp_id = 'GCP-%s-%s' % (x, y)
            gcps.append(gdal.GCP(lon, lat, 0.0, x, y, gcp_id, gcp_id))

    mem_driver = gdal.GetDriverByName("MEM")

    lon1 = lon_min
    lon2 = lon_min + res * dst_width
    lat1 = lat_max - res * dst_height
    lat2 = lat_max

    dst_dataset = xr.Dataset()
    dst_dataset['lon_bnds'] = (['lon', 'bnds'],
                               # TODO: find more elegant numpy expr for the following
                               list(zip(np.linspace(lon1, lon2 - res, dst_width),
                                        np.linspace(lon1 + res, lon2, dst_width))),
                               dict(units='degrees_east'))
    dst_dataset['lat_bnds'] = (['lat', 'bnds'],
                               # TODO: find more elegant numpy expr for the following
                               list(zip(np.linspace(lat2, lat1 + res, dst_height),
                                        np.linspace(lat2 - res, lat1, dst_height))),
                               dict(units='degrees_north'))
    dst_dataset.coords['lon'] = (['lon', ],
                                 np.linspace(lon1 + res / 2, lon2 - res / 2, dst_width),
                                 dict(bounds='lon_bnds', long_name='longitude', units='degrees_east'))
    dst_dataset.coords['lat'] = (['lat', ],
                                 np.linspace(lat2 - res / 2, lat1 + res / 2, dst_height),
                                 dict(bounds='lat_bnds', long_name='latitude', units='degrees_north'))
    dst_dataset.attrs = dataset.attrs

    for var_name in dataset.variables:
        src_var = dataset[var_name]

        if var_name == 'lat' or var_name == 'lon':
            # Don't store lat and lon 2D vars in destination
            continue

        if src_var.dims != lon_var.dims:
            # Store any variable as-is, that does not have the lat/lon 2D dims, then continue
            dst_dataset[var_name] = src_var
            continue

        src_width = src_var.shape[-1]
        src_height = src_var.shape[-2]

        # TODO: collect variables of same type and set band_count accordingly to speed up reprojection from GCPs
        band_count = 1
        # TODO: select the data_type based on src_var.dtype
        data_type = gdal.GDT_Float64

        src_var_dataset = mem_driver.Create('src_' + var_name, src_width, src_height, band_count, data_type, [])
        src_var_dataset.SetGCPs(gcps, CRS_WKT_EPSG_4326)

        dst_var_dataset = mem_driver.Create('dst_' + var_name, dst_width, dst_height, band_count, data_type, [])
        dst_var_dataset.SetProjection(CRS_WKT_EPSG_4326)
        dst_var_dataset.SetGeoTransform(dst_geo_transform)

        for band_index in range(1, band_count + 1):
            src_var_dataset.GetRasterBand(band_index).SetNoDataValue(float('nan'))
            src_var_dataset.GetRasterBand(band_index).WriteArray(src_var.values)
            dst_var_dataset.GetRasterBand(band_index).SetNoDataValue(float('nan'))

        # resampling = gdal.GRA_Bilinear
        # TODO: configure resampling individually for each variable
        resampling = gdal.GRA_NearestNeighbour
        warp_mem_limit = 0
        error_threshold = 0
        # TODO: configure no-data value to be used for outside regions = float('nan'), it is 0 by default --> wrong!
        options = []
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
        # print(var_name, dst_values.shape, dst_values.min(), dst_values.max())

        # TODO: set CF-1.6 attributes correctly
        # TODO: set encoding to compress and optimize output: scaling_factor, add_offset, _FillValue, chunksizes
        dst_dataset[var_name] = xr.DataArray(dst_values, dims=['lat', 'lon'], name=var_name, attrs=src_var.attrs)

    return dst_dataset
