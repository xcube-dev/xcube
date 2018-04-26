from typing import Tuple

import gdal
import xarray as xr
import numpy as np

from .constants import CRS_WKT_EPSG_4326

GeoRange = Tuple[float, float, float, float]

# TODO: add variables to be included/excluded
# TODO: add src and dst CRS
def reproject_xarray(dataset: xr.Dataset,
                     dst_width: int,
                     dst_height: int,
                     region: GeoRange = None,
                     gcp_step: int = 10,
                     tp_gcp_step: int = 1) -> xr.Dataset:
    assert dataset is not None
    assert dst_width > 1
    assert dst_height > 1
    assert gcp_step > 0

    lon_var = dataset.lon
    assert len(lon_var.dims) == 2
    assert lon_var.shape[-1] >= 2
    assert lon_var.shape[-2] >= 2

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
    gcp_index = 0
    for y in np.linspace(0, src_height - 1, gcp_step, dtype=np.int32):
        for x in np.linspace(0, src_width - 1, gcp_step, dtype=np.int32):
            lon, lat = float(lon_values[y, x]), float(lat_values[y, x])
            gcps.append(gdal.GCP(lon, lat, 0.0, x + 0.5, y + 0.5, '%s,%s' % (x, y), str(gcp_index)))
            gcp_index += 1

    # TODO: turn following names into parameters
    tp_x_name = 'tp_x'
    tp_y_name = 'tp_y'
    tp_lon_name = 'TP_longitude'
    tp_lat_name = 'TP_latitude'
    if tp_x_name in dataset.sizes and tp_y_name in dataset.sizes \
            and tp_lon_name in dataset and tp_lat_name in dataset:
        tp_gcps = []
        tp_width = dataset.sizes[tp_x_name]
        tp_height = dataset.sizes[tp_y_name]
        tp_lon_var = dataset[tp_lon_name]
        tp_lat_var = dataset[tp_lat_name]
        tp_lon_values = tp_lon_var.values
        tp_lat_values = tp_lat_var.values
        gcp_index = 0
        for y in np.linspace(0, tp_height - 1, tp_gcp_step, dtype=np.int32):
            for x in np.linspace(0, tp_width - 1, tp_gcp_step, dtype=np.int32):
                lon, lat = float(tp_lon_values[y, x]), float(tp_lat_values[y, x])
                tp_gcps.append(gdal.GCP(lon, lat, 0.0, x + 0.5, y + 0.5, '%s,%s' % (x, y), str(gcp_index)))
                gcp_index += 1
    else:
        tp_gcps = None
        tp_width = None
        tp_height = None
        tp_lon_var = None

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
            # TODO: instead add lat and lon with new names, so we can validate geo-location
            continue

        if src_var.dims == lon_var.dims:

            # TODO: collect variables of same type and size and set band_count accordingly to speed up reprojection
            band_count = 1
            # TODO: select the data_type based on src_var.dtype
            data_type = gdal.GDT_Float64

            src_var_dataset = mem_driver.Create('src_' + var_name, src_width, src_height, band_count, data_type, [])
            src_var_dataset.SetGCPs(gcps, CRS_WKT_EPSG_4326)

        elif tp_lon_var is not None and src_var.dims == tp_lon_var.dims:
            if var_name == tp_lat_name or var_name == tp_lon_name:
                # Don't store TP lat and TP lon 2D vars in destination
                # TODO: instead add TP lat and lon with new names, so we can validate geo-location
                continue

            # TODO: collect variables of same type and size and set band_count accordingly to speed up reprojection
            band_count = 1
            # TODO: select the data_type based on src_var.dtype
            data_type = gdal.GDT_Float64

            src_var_dataset = mem_driver.Create('src_' + var_name, tp_width, tp_height, band_count, data_type, [])
            src_var_dataset.SetGCPs(tp_gcps, CRS_WKT_EPSG_4326)
        else:
            # Store any variable as-is, that does not have the lat/lon 2D dims, then continue
            print('storeded as-is: ', var_name, src_var.sizes, src_var.min(), src_var.max())
            dst_dataset[var_name] = src_var
            continue

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
        print(var_name, dst_values.shape, dst_values.min(), dst_values.max())

        # TODO: set CF-1.6 attributes correctly
        # TODO: set encoding to compress and optimize output: scaling_factor, add_offset, _FillValue, chunksizes
        dst_dataset[var_name] = xr.DataArray(dst_values, dims=['lat', 'lon'], name=var_name, attrs=src_var.attrs)

    return dst_dataset
