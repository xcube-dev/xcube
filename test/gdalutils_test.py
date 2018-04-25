import os
import unittest

import xarray as xr
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt

from xcube.gdalutils import reproject_xarray

# HIGHROC_NC = "D:\\EOData\\HIGHROC\\0001_SNS\\OLCI\\2017\\04\\O_L2_0001_SNS_2017104102450_v1.0.nc"
HIGHROC_NC = "D:\\BC\\HIGHROC\\O_L2_0001_SNS_2017104102450_v1.0.nc"


class GdalUtilsTest(unittest.TestCase):

    def test_reproject_xarray(self):
        if not os.path.isfile(HIGHROC_NC):
            print('warning: test_reproject_xarray() not executed')

        dst_width = 1024
        dst_height = 512

        dataset = xr.open_dataset(HIGHROC_NC, decode_cf=True, decode_coords=False)
        new_dataset = reproject_xarray(dataset, dst_width, dst_height, gcp_step=50)
        self.assertIsNotNone(new_dataset)

        _rm('highroc-test-out.nc')
        new_dataset.to_netcdf('highroc-test-out.nc')

        # for var_name in new_dataset.variables:
        #     var = new_dataset[var_name]
        #     if var.dims == ('lat', 'lon'):
        #         var.plot()


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
