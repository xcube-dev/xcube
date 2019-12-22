import time
import unittest

import numpy as np
import xarray as xr

from xcube.core.geocoded import compute_output_geom, reproject

TEST_INPUT = '../xcube-gen-bc/test/inputdata/O_L2_0001_SNS_2017104102450_v1.0.nc'


class ReprojectTest(unittest.TestCase):
    def test_compute_output_geom(self):
        src_ds = xr.open_dataset(TEST_INPUT)

        width, height, lon_min, lat_min, res = compute_output_geom(src_ds)
        self.assertEqual(834, width)
        self.assertEqual(535, height)
        self.assertAlmostEqual(1.152138, lon_min, delta=1e-5)
        self.assertAlmostEqual(52.32967, lat_min, delta=1e-5)
        self.assertAlmostEqual(0.0005832, res, delta=1e-6)

        width, height, lon_min, lat_min, res = compute_output_geom(src_ds, denom_x=4, denom_y=2)
        self.assertEqual(836, width)
        self.assertEqual(536, height)
        self.assertAlmostEqual(1.152138, lon_min, delta=1e-5)
        self.assertAlmostEqual(52.32967, lat_min, delta=1e-5)
        self.assertAlmostEqual(0.0005832, res, delta=1e-6)


    def test_reproject(self):
        src_ds = xr.open_dataset(TEST_INPUT)
        src_var = src_ds.rtoa_8

        dst_width, dst_height, lon_min, lat_max, dst_res = compute_output_geom(src_ds)

        dst_var_values = np.full((dst_height, dst_width), np.nan, dtype=src_var.dtype)
        t1 = time.perf_counter()
        reproject(src_var.values,
                  src_ds.lon.values,
                  src_ds.lat.values,
                  dst_var_values,
                  lon_min,
                  lat_max,
                  dst_res)
        t2 = time.perf_counter()
        print('time:', t2 - t1)

        self.assertEqual(301623, np.sum(np.where(np.isnan(dst_var_values), 0, 1)))

        self.assertAlmostEqual(float(src_var.min()),
                               float(np.nanmin(dst_var_values)),
                               delta=1e-10)

        self.assertAlmostEqual(float(src_var.max()),
                               float(np.nanmax(dst_var_values)),
                               delta=1e-10)

        self.assertAlmostEqual(float(src_var.mean()),
                               float(np.nanmean(dst_var_values)),
                               delta=1e-2)

        lon_start = lon_min + 0.5 * dst_res
        lat_start = lat_max + 0.5 * dst_res
        dst_var = xr.DataArray(dst_var_values,
                               dims=('lat', 'lon'),
                               coords=dict(lon=np.linspace(lon_start,
                                                           lon_start + (dst_width - 1) * dst_res,
                                                           num=dst_width,
                                                           dtype=np.float64),
                                           lat=np.linspace(lat_start,
                                                           lat_start + (dst_height - 1) * dst_res,
                                                           num=dst_height,
                                                           dtype=np.float64)),
                               attrs=src_var.attrs)
        dst_ds = xr.Dataset({src_var.name: dst_var})
        dst_ds.to_netcdf('reprojected.nc')
