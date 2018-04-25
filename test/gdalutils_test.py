import os.path
import unittest

import gdal
import numpy as np
import xarray as xr

from xcube.constants import CRS_WKT_EPSG_4326
from xcube.gdalutils import reproject, open_netcdf_with_gcps

HIGHROC_NC = "D:\\EOData\\HIGHROC\\0001_SNS\\OLCI\\2017\\04\\O_L2_0001_SNS_2017104102450_v1.0.nc"


class GdalUtilsTest(unittest.TestCase):

    def test_highroc_gdal_gcps(self):
        if not os.path.isfile(HIGHROC_NC):
            print('warning: test_highroc_gdal_gcps() not executed')

        dst_width = 512
        dst_height = 256

        dataset, lon_dataset, lat_dataset, var_datasets = open_netcdf_with_gcps(HIGHROC_NC, gcp_step=50)

        src_size = (lon_dataset.GetRasterBand(1).XSize,
                    lon_dataset.GetRasterBand(1).YSize)
        print('src_size =', src_size)
        # src_size = (1142, 472)

        src_lon_range = lon_dataset.GetRasterBand(1).ComputeRasterMinMax()
        print('src_lon_range =', src_lon_range)
        # src_lat_range = (0.7709564552077518, 5.598963445135737)

        src_lat_range = lat_dataset.GetRasterBand(1).ComputeRasterMinMax()
        print('src_lat_range =', src_lat_range)
        # src_lat_range = (50.697340411487794, 52.641403823455065)

        src_lon_min, src_lon_max = src_lon_range
        src_lat_min, src_lat_max = src_lat_range

        extra_degrees = 0.1 * (src_lon_max - src_lon_min)
        dst_lon_0 = src_lon_min - extra_degrees / 2
        dst_lat_0 = src_lat_min - extra_degrees / 2
        dst_res = (src_lon_max - src_lon_min + extra_degrees) / dst_width
        dst_geo_transform = (dst_lon_0, dst_res, 0.0,
                             dst_lat_0, 0.0, dst_res)
        dst_projection = CRS_WKT_EPSG_4326

        for var_name, var_dataset in var_datasets.items():
            reproject(var_name, var_dataset, dst_width, dst_height, dst_geo_transform, dst_projection)

    def test_gdal_gcps(self):
        _rm('test.nc')
        _rm('test.nc.aux.xml')

        width = 50
        height = 100
        lon_min = 10.
        lon_max = 14.
        lat_min = 50.
        lat_max = 58.

        _write_test_dataset('test.nc',
                            width=width, height=height,
                            lon_min=lon_min, lat_min=lat_min,
                            lon_max=lon_max, lat_max=lat_max,
                            geo_non_linearity=True)

        dataset, lon_dataset, lat_dataset, var_datasets = open_netcdf_with_gcps('test.nc')
        self.assertIn('iop_apig', var_datasets)
        self.assertIn('iop_atot', var_datasets)

        extra_degrees = 4.
        dst_lon_0 = lon_min - extra_degrees / 2
        dst_lat_0 = lat_min - extra_degrees / 2
        dst_res = (lon_max - lon_min + extra_degrees) / width
        dst_geo_transform = (dst_lon_0, dst_res, 0.0,
                             dst_lat_0, 0.0, dst_res)
        dst_projection = CRS_WKT_EPSG_4326

        for var_name, var_dataset in var_datasets.items():
            reproject(var_name, var_dataset, width, height, dst_geo_transform, dst_projection)

    def test_gdal(self):
        _rm('test.nc')
        _rm('test.nc.aux.xml')
        _write_test_dataset('test.nc')

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
        _rm('test.nc')
        _rm('test.nc.aux.xml')


def _write_test_dataset(path,
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
        lon_data = _apply_and_norm(lon_data, lambda a: np.square(np.square(a)))
        lat_data = _apply_and_norm(lat_data, lambda a: np.square(np.square(a)))

    dataset = xr.Dataset({'iop_apig': (('y', 'x'),
                                       iop_apig_data,
                                       dict(units='m^-1',
                                            long_name='Absorption coefficient of phytoplankton pigments',
                                            _FillValue=-1.)),
                          'iop_atot': (('y', 'x'),
                                       iop_atot_data,
                                       dict(units='m^-1',
                                            long_name='phytoplankton + detritus + gelbstoff absorption',
                                            _FillValue=-1.))
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


def _rm(path):
    import os
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except:
            pass


def _apply_and_norm(data, f):
    old_min = data.min()
    old_max = data.max()
    new_data = f(data)
    new_min = new_data.min()
    new_max = new_data.max()
    return old_min + (old_max - old_min) * (new_data - new_min) / (new_max - new_min)
