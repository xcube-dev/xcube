import struct
import unittest

import numpy as np
import xarray as xr
import gdal

EPSG_4326 = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]
"""


class WarpTest(unittest.TestCase):

    def test_gdal_gcps(self):
        rm('test.nc')
        rm('test.nc.aux.xml')

        width = 50
        height = 100
        lon_min = 10.
        lon_max = 14.
        lat_min = 50.
        lat_max = 58.
        write_test_dataset('test.nc', width=50, height=100,
                           lon_min=lon_min, lat_min=lat_min,
                           lon_max=lon_max, lat_max=lat_max,
                           geo_non_linearity=True)

        extra = 4.
        dst_lon_0 = lon_min - extra / 2
        dst_lat_0 = lat_min - extra / 2
        dst_res = (lon_max - lon_min + extra) / width
        dst_geo_transform = (dst_lon_0, dst_res, 0.0,
                             dst_lat_0, 0.0, dst_res)
        dst_projection = EPSG_4326

        dataset, var_datasets = open_netcdf_with_gcps('test.nc')
        self.assertIn('iop_apig', var_datasets)
        self.assertIn('iop_atot', var_datasets)

        for var_name, var_dataset in var_datasets.items():

            src_band = var_dataset.GetRasterBand(1)
            src_no_data_value = src_band.GetNoDataValue()

            dst_driver = gdal.GetDriverByName('MEM')
            dst_dataset = dst_driver.Create(var_name,
                                            var_dataset.RasterXSize,
                                            var_dataset.RasterYSize,
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
            warp_mem_limit = 1000000
            error_threshold = 0.125  # error threshold (in degree?)
            options = []

            callback_data = dict()
            gdal.ReprojectImage(var_dataset,
                                dst_dataset,
                                None,
                                dst_projection,
                                resampling,
                                warp_mem_limit,
                                error_threshold,
                                callback,
                                callback_data,
                                options)  # options

            dst_data = dst_band.ReadAsArray()
            print('dst_data min/max =', dst_data.min(), dst_data.max())

    def ____test_gdal(self):
        rm('test.nc')
        rm('test.nc.aux.xml')
        write_test_dataset('test.nc')

        dataset = gdal.Open('test.nc')
        self.assertIsNotNone(dataset)
        print('dataset =', dataset)

        sub_datasets = dataset.GetSubDatasets()
        self.assertIsNotNone(sub_datasets)
        self.assertEqual(len(sub_datasets), 4)
        self.assertEqual(sub_datasets[0], ('HDF5:"test.nc"://iop_apig', '[3x4] //iop_apig (64-bit floating-point)'))
        self.assertEqual(sub_datasets[1], ('HDF5:"test.nc"://iop_atot', '[3x4] //iop_atot (64-bit floating-point)'))
        self.assertEqual(sub_datasets[2], ('HDF5:"test.nc"://lat', '[3x4] //lat (64-bit floating-point)'))
        self.assertEqual(sub_datasets[3], ('HDF5:"test.nc"://lon', '[3x4] //lon (64-bit floating-point)'))
        print('sub_datasets =', sub_datasets)

        sub_dataset = gdal.Open(sub_datasets[0][0])
        self.assertIsNotNone(sub_dataset)
        print('subdataset =', sub_dataset)

        gcp_projection = sub_dataset.GetGCPProjection()
        self.assertEqual(gcp_projection, '')
        print('gcp_projection =', gcp_projection)

        gcps = sub_dataset.GetGCPs()
        self.assertIsNotNone(gcps)
        self.assertEqual(len(gcps), 0)
        print('gcps =', gcps)

        geo_transform = sub_dataset.GetGeoTransform()
        self.assertIsNotNone(geo_transform)
        print('geo_transform =', geo_transform)

        # gcps = []
        # for j in range(y_size):
        #     for i in range(x_size):
        #         lon = lon_data[j][i]
        #         lat = lon_data[j][i]
        #         gcp_id = 'GCP-%s-%s' % (i, j)
        #         gcps.append(gdal.GCP(lon, lat, 0.0, i, j, gcp_id, gcp_id))
        #
        # sub_dataset.SetGCPs(gcps, EPSG_4326)
        #
        # gcp_projection = sub_dataset.GetGCPProjection()
        # self.assertEqual(gcp_projection, EPSG_4326)
        # print('gcp_projection =', gcp_projection)

        # Close sub_dataset
        sub_dataset = None
        # Close dataset
        dataset = None

        # ds = gdal.Translate('test-out.nc', ds, projWin=[-75.3, 5.5, -73.5, 3.7])

        # print(ds)
        # https://gis.stackexchange.com/questions/233375/sentinel-1-data-opened-with-rasterio-has-no-affine-transform-crs?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        rm('test.nc')
        rm('test.nc.aux.xml')


def open_netcdf_with_gcps(path):
    dataset = gdal.Open(path)
    assert dataset is not None

    sub_datasets_infos = dataset.GetSubDatasets()
    assert sub_datasets_infos is not None

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
    struct_fmt = 'd' * lon_band.XSize
    gcps = []
    for y in range(lon_band.YSize):
        lon_scanline = lon_band.ReadRaster(xoff=0, yoff=0,
                                           xsize=lon_band.XSize, ysize=1,
                                           buf_xsize=lon_band.XSize, buf_ysize=1,
                                           buf_type=gdal.GDT_Float64)
        lon_data = struct.unpack(struct_fmt, lon_scanline)
        lat_scanline = lat_band.ReadRaster(xoff=0, yoff=0,
                                           xsize=lat_band.XSize, ysize=1,
                                           buf_xsize=lat_band.XSize, buf_ysize=1,
                                           buf_type=gdal.GDT_Float64)
        lat_data = struct.unpack(struct_fmt, lat_scanline)
        for x in range(lon_band.XSize):
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

        var_dataset.SetGCPs(gcps, EPSG_4326)
        var_dataset.SetGeoTransform((0., 1., 0., 0., 0., 1.))
        var_datasets[var_name[var_name.rindex('/') + 1:]] = var_dataset

    lon_dataset = None
    lat_dataset = None

    return dataset, var_datasets


def write_test_dataset(path,
                       width=10, height=10,
                       lon_min=-10., lat_min=-10., lon_max=10., lat_max=10.,
                       geo_non_linearity=False):
    x = np.linspace(1, 2, width)
    y = np.linspace(3, 4, height)
    iop_apig_data, iop_atot_data = np.meshgrid(x, y, sparse=False)

    x = np.linspace(lon_min, lon_max, width)
    y = np.linspace(lat_min, lat_max, height)
    lon_data, lat_data = np.meshgrid(x, y, sparse=False)

    if geo_non_linearity:
        lon_data = apply_and_norm(lon_data, np.square)
        lat_data = apply_and_norm(lat_data, np.sqrt)

    dataset = xr.Dataset({'iop_apig': (('y', 'x'),
                                       iop_apig_data,
                                       dict(units='m^-1',
                                            long_name='Absorption coefficient of phytoplankton pigments')),
                          'iop_atot': (('y', 'x'),
                                       iop_atot_data,
                                       dict(units='m^-1',
                                            long_name='phytoplankton + detritus + gelbstoff absorption'))
                          },
                         {'lat': (('y', 'x'), lat_data,
                                  dict(standard_name='latitude', long_name='latitude', units='degrees')),
                          'lon': (('y', 'x'), lon_data,
                                  dict(standard_name='longitude', long_name='longitude', units='degrees'))
                          },
                         dict(start_date='2014-05-01T10:00:00',
                              stop_date='2014-05-01T10:02:30'))

    dataset.to_netcdf(path)
    dataset.close()


def rm(path):
    import os
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except:
            pass


def apply_and_norm(data, f):
    old_min = data.min()
    old_max = data.max()
    new_data = f(data)
    new_min = new_data.min()
    new_max = new_data.max()
    return old_min + (old_max - old_min) * (new_data - new_min) / (new_max - new_min)
