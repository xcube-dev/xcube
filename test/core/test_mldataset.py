import os
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.mldataset import BaseMultiLevelDataset, ComputedMultiLevelDataset
from xcube.util.tilegrid import TileGrid


class BaseMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ds = _get_test_dataset()

        ml_ds = BaseMultiLevelDataset(ds)
        self.assertEqual(3, ml_ds.num_levels)
        self.assertEqual(TileGrid(3, 2, 1, 180, 180, (-180, -90, 180, 90), inv_y=False),
                         ml_ds.tile_grid)

        ds0 = ml_ds.get_dataset(0)
        self.assertIs(ds, ds0)

        ds1 = ml_ds.get_dataset(1)
        self.assertIsNot(ds, ds1)
        self.assertEqual({'time': 14, 'lat': 360, 'lon': 720}, ds1.dims)

        ds2 = ml_ds.get_dataset(2)
        self.assertIsNot(ds, ds2)
        self.assertEqual({'time': 14, 'lat': 180, 'lon': 360}, ds2.dims)

        ml_ds.close()


class ComputedMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ds = _get_test_dataset()

        ml_ds1 = BaseMultiLevelDataset(ds)

        def input_ml_dataset_getter(ds_id):
            if ds_id == "ml_ds1":
                return ml_ds1
            self.fail(f"unexpected ds_id={ds_id!r}")

        ml_ds2 = ComputedMultiLevelDataset("ml_ds2",
                                           os.path.join(os.path.dirname(__file__),
                                                        "..", "webapi", "res", "test", "script.py"),
                                           "compute_dataset",
                                           ["ml_ds1"],
                                           input_ml_dataset_getter,
                                           input_parameters=dict(period='1W'))
        self.assertEqual(3, ml_ds2.num_levels)
        self.assertEqual(TileGrid(3, 2, 1, 180, 180, (-180, -90, 180, 90), inv_y=False),
                         ml_ds2.tile_grid)

        ds0 = ml_ds2.get_dataset(0)
        self.assertEqual({'time': 3, 'lat': 720, 'lon': 1440, 'bnds': 2}, ds0.dims)

        ds1 = ml_ds2.get_dataset(1)
        self.assertEqual({'time': 3, 'lat': 360, 'lon': 720}, ds1.dims)

        ds2 = ml_ds2.get_dataset(2)
        self.assertEqual({'time': 3, 'lat': 180, 'lon': 360}, ds2.dims)

        ml_ds1.close()
        ml_ds2.close()


def _get_test_dataset():
    w = 1440
    h = 720
    p = 14

    x1 = -180.
    y1 = -90.
    x2 = +180.
    y2 = +90.
    dx = (x2 - x1) / w
    dy = (y2 - y1) / h

    data_vars = dict(noise=(("time", "lat", "lon"), np.random.rand(p, h, w)))

    lat_bnds = np.array(list(zip(np.linspace(y2, y1 + dy, num=h),
                                 np.linspace(y2 - dy, y1, num=h))))
    lon_bnds = np.array(list(zip(np.linspace(x1, x2 - dx, num=w),
                                 np.linspace(x1 + x2, x2, num=w))))
    coords = dict(time=(("time",),
                        np.array(pd.date_range(start="2019-01-01T12:00:00Z", periods=p, freq="1D"),
                                 dtype="datetime64[ns]")),
                  lat=(("lat",),
                       np.linspace(y2 - .5 * dy, y1 + .5 * dy, num=h)),
                  lon=(("lon",),
                       np.linspace(x1 + .5 * dx, x2 - .5 * dx, num=w)),
                  lat_bnds=(("lat", "bnds"), lat_bnds),
                  lon_bnds=(("lon", "bnds"), lon_bnds))

    return xr.Dataset(coords=coords, data_vars=data_vars)
