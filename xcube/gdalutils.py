import struct
import math

import gdal

from .constants import CRS_WKT_EPSG_4326

def reproject(var_name, var_dataset, dst_width, dst_height, dst_geo_transform, dst_projection):
    src_band = var_dataset.GetRasterBand(1)
    src_no_data_value = src_band.GetNoDataValue()
    dst_driver = gdal.GetDriverByName('MEM')
    dst_dataset = dst_driver.Create(var_name,
                                    dst_width,
                                    dst_height,
                                    1,
                                    gdal.GDT_Float64)
    dst_dataset.SetGeoTransform(dst_geo_transform)
    dst_dataset.SetProjection(dst_projection)
    dst_band = dst_dataset.GetRasterBand(1)
    dst_band.SetNoDataValue(float('nan') if src_no_data_value is None else src_no_data_value)

    def callback(callback_data: dict):
        callback_data['worked'] = 1
        print('progress!')
        return True

    # resampling = gdal.GRA_Bilinear
    resampling = gdal.GRA_NearestNeighbour
    warp_mem_limit = 0
    error_threshold = 0
    options = []
    callback_data = dict()
    gdal.ReprojectImage(var_dataset,
                        dst_dataset,
                        None,
                        dst_projection,
                        resampling,
                        warp_mem_limit,
                        error_threshold,
                        None,  # callback,
                        None,  # callback_data,
                        options)  # options
    dst_data = dst_band.ReadAsArray()
    from matplotlib import pyplot as plt
    plt.imshow(dst_data)
    plt.show()
    # print('dst_data min/max =', dst_data.min(), dst_data.max())


def open_netcdf_with_gcps(path, gcp_step=1):
    dataset = gdal.Open(path)
    assert dataset is not None

    sub_datasets_infos = dataset.GetSubDatasets()
    assert sub_datasets_infos is not None

    #mem_driver = gdal.GetDriverByName("MEM")
    #mem_dataset = mem_driver.CreateCopy('mem_dataset', dataset, 0)
    #sub_datasets_infos = mem_dataset.GetSubDatasets()
    #assert sub_datasets_infos is not None

    lon_name = None
    lat_name = None
    var_names = []
    for sub_dataset_name, _ in sub_datasets_infos:
        if sub_dataset_name.endswith('/lon'):
            lon_name = sub_dataset_name
        elif sub_dataset_name.endswith('/lat'):
            lat_name = sub_dataset_name
        elif sub_dataset_name.endswith('/time'):
            # TODO
            pass
        else:
            var_names.append(sub_dataset_name)

    if lon_name and lat_name:
        lon_dataset = gdal.Open(lon_name)
        assert lon_dataset is not None
        lat_dataset = gdal.Open(lat_name)
        assert lat_dataset is not None
    else:
        raise ValueError('both "lon" and "lat" datasets must be given')

    def same_size(ds1, ds2):
        return ds1.RasterXSize == ds2.RasterXSize and ds1.RasterYSize == ds2.RasterYSize

    if not same_size(lon_dataset, lat_dataset):
        raise ValueError('both "lon" and "lat" must have the same shape')

    lon_band = lon_dataset.GetRasterBand(1)
    lat_band = lat_dataset.GetRasterBand(1)
    width, height = lon_band.XSize, lon_band.YSize
    struct_fmt = 'd' * width
    gcps = []
    for y in range(0, lon_band.YSize, gcp_step):
        lon_scanline = lon_band.ReadRaster(xoff=0, yoff=y,
                                           xsize=lon_band.XSize, ysize=1,
                                           buf_xsize=lon_band.XSize, buf_ysize=1,
                                           buf_type=gdal.GDT_Float64)
        lon_data = struct.unpack(struct_fmt, lon_scanline)
        lat_scanline = lat_band.ReadRaster(xoff=0, yoff=y,
                                           xsize=lat_band.XSize, ysize=1,
                                           buf_xsize=lat_band.XSize, buf_ysize=1,
                                           buf_type=gdal.GDT_Float64)
        lat_data = struct.unpack(struct_fmt, lat_scanline)
        for x in range(0, lon_band.XSize, gcp_step):
            lon, lat = lon_data[x], lat_data[x]
            gcp_id = 'GCP-%s-%s' % (x, y)
            gcps.append(gdal.GCP(lon, lat, 0.0, x, y, gcp_id, gcp_id))

    var_datasets = {}
    for var_name in var_names:
        var_dataset = gdal.Open(var_name)
        assert var_dataset is not None
        if not same_size(lon_dataset, var_dataset):
            # Close var_dataset
            del var_dataset
            continue

        var_dataset.SetGCPs(gcps, CRS_WKT_EPSG_4326)
        var_dataset.SetGeoTransform((0., 1., 0., 0., 0., 1.))
        var_datasets[var_name[var_name.rindex('/') + 1:]] = var_dataset

        var_band = var_dataset.GetRasterBand(1)
        var_no_data_value = var_band.GetNoDataValue()
        print(var_name, var_no_data_value)
        # var_range = var_band.ComputeRasterMinMax(False)
        # print(var_name, var_range)

    return dataset, lon_dataset, lat_dataset, var_datasets
