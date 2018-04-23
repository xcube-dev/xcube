import unittest

import numpy as np
import xarray as xr


class WarpTest(unittest.TestCase):

    def test_warp(self):
        dims = 'y', 'x'
        global_attribs = dict(start_date='2014-05-01T10:00:00', stop_date='2014-05-01T10:02:30')
        y_size, x_size = 10, 20
        iop_apig_data = np.random.random_sample((y_size, x_size))
        iop_atot_data = np.random.random_sample((y_size, x_size))
        lat_data = [[30., 30., 30., 30.],
                    [20., 20., 20., 20.],
                    [10., 10., 10., 10.]]
        lon_data = [[-10., 0., 10., 20.],
                    [-10., 0., 10., 20.],
                    [-10., 0., 10., 20.]]
        dataset = xr.Dataset({'iop_apig ': (dims, iop_apig_data, dict(units='m^-1',
                                                                      long_name='Absorption coefficient of phytoplankton pigments')),
                              'iop_atot': (dims, iop_atot_data, dict(units='m^-1',
                                                                     long_name='phytoplankton + detritus + gelbstoff absorption'))
                              },
                             {'lat': (('y', 'x'), lat_data),
                              'lon': (('y', 'x'), lon_data)
                              },
                             global_attribs)
        # https://gis.stackexchange.com/questions/233375/sentinel-1-data-opened-with-rasterio-has-no-affine-transform-crs?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa