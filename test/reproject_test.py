import os
import unittest

import xarray as xr
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt

from xcube.reproject import reproject_to_wgs84

HIGHROC_NC = "D:\\EOData\\HIGHROC\\0001_SNS\\OLCI\\2017\\04\\O_L2_0001_SNS_2017104102450_v1.0.nc"


# HIGHROC_NC = "D:\\BC\\HIGHROC\\O_L2_0001_SNS_2017104102450_v1.0.nc"


class ReprojectTest(unittest.TestCase):

    def test_reproject_to_wgs84(self):
        if not os.path.isfile(HIGHROC_NC):
            print('warning: test_reproject_xarray() not executed')

        dst_width = 1024
        dst_height = 512

        dataset = xr.open_dataset(HIGHROC_NC, decode_cf=True, decode_coords=False)
        new_dataset = reproject_to_wgs84(dataset, dst_width, dst_height, gcp_i_step=50)
        self.assertIsNotNone(new_dataset)
        self.assertEquals(new_dataset.sizes, dict(lon=dst_width, lat=dst_height, time=1, bnds=2))
        self.assertIn('lon', new_dataset)
        self.assertIn('lon_bnds', new_dataset)
        self.assertIn('lat', new_dataset)
        self.assertIn('lat_bnds', new_dataset)
        self.assertIn('time', new_dataset)
        self.assertIn('time_bnds', new_dataset)
        self.assertEqual(new_dataset.lon.shape, (dst_width,))
        self.assertEqual(new_dataset.lon_bnds.shape, (dst_width, 2))
        self.assertEqual(new_dataset.lat.shape, (dst_height,))
        self.assertEqual(new_dataset.lat_bnds.shape, (dst_height, 2))
        self.assertEqual(new_dataset.time.shape, (1,))
        self.assertEqual(new_dataset.time_bnds.shape, (1, 2))

        # _rm('highroc-test-out.nc')
        # new_dataset.to_netcdf('highroc-test-out.nc')

        # from matplotlib import pyplot as plt
        # for var_name in new_dataset.variables:
        #     var = new_dataset[var_name]
        #     if var.dims == ('lat', 'lon'):
        #         var.plot()
        #         plt.show()


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
