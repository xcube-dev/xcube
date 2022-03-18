import os
import unittest

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from test.s3test import S3Test, MOTO_SERVER_ENDPOINT_URL
from xcube.core.dsio import write_dataset
from xcube.core.geom import clip_dataset_by_geometry
from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import CombinedMultiLevelDataset
from xcube.core.mldataset import ComputedMultiLevelDataset
from xcube.core.mldataset import ObjectStorageMultiLevelDataset
from xcube.core.mldataset import open_ml_dataset_from_object_storage
from xcube.core.mldataset import write_levels
from xcube.util.tilegrid import ImageTileGrid


class CombinedMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ml_ds_1 = BaseMultiLevelDataset(
            _get_test_dataset(('noise_1', 'noise_2'))
        )
        ml_ds_2 = BaseMultiLevelDataset(
            _get_test_dataset(('noise_3', 'noise_4'))
        )
        ml_ds_3 = BaseMultiLevelDataset(
            _get_test_dataset(('noise_5', 'noise_6'))
        )

        ml_ds = CombinedMultiLevelDataset([ml_ds_1, ml_ds_2, ml_ds_3])

        self.assertEqual(3, ml_ds.num_levels)
        self.assertIsInstance(ml_ds.tile_grid, ImageTileGrid)
        self.assertEqual((180, 180), ml_ds.tile_grid.tile_size)
        self.assertEqual(3, ml_ds.tile_grid.num_levels)

        expected_var_names = {'noise_1', 'noise_2',
                              'noise_3', 'noise_4',
                              'noise_5', 'noise_6'}

        ds0 = ml_ds.get_dataset(0)
        self.assertEqual({'time': 14, 'lat': 720, 'lon': 1440, 'bnds': 2},
                         ds0.dims)
        self.assertEqual(expected_var_names, set(map(str, ds0.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon')
                            for v in ds0.data_vars.values()))

        ds1 = ml_ds.get_dataset(1)
        self.assertEqual({'time': 14, 'lat': 360, 'lon': 720},
                         ds1.dims)
        self.assertEqual(expected_var_names, set(map(str, ds1.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon')
                            for v in ds1.data_vars.values()))

        ds2 = ml_ds.get_dataset(2)
        self.assertEqual({'time': 14, 'lat': 180, 'lon': 360},
                         ds2.dims)
        self.assertEqual(expected_var_names, set(map(str, ds2.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon')
                            for v in ds2.data_vars.values()))

        self.assertEqual([ds0, ds1, ds2], ml_ds.datasets)

        ml_ds.close()


class BaseMultiLevelDatasetTest(unittest.TestCase):
    def test_basic_props(self):
        ds = _get_test_dataset()
        ml_ds = BaseMultiLevelDataset(ds)

        self.assertIsInstance(ml_ds.ds_id, str)

        self.assertEqual(3, ml_ds.num_levels)
        self.assertEqual([(0.25, 0.25), (0.5, 0.5), (1.0, 1.0)],
                         ml_ds.resolutions)

        tile_grid = ml_ds.tile_grid
        self.assertIsInstance(tile_grid, ImageTileGrid)
        self.assertEqual((180, 180), tile_grid.tile_size)
        self.assertEqual(3, tile_grid.num_levels)

    def test_level_datasets(self):
        ds = _get_test_dataset()
        ml_ds = BaseMultiLevelDataset(ds)

        ds0 = ml_ds.get_dataset(0)
        self.assertIsNot(ds, ds0)
        self.assertEqual({'time': 14, 'lat': 720, 'lon': 1440, 'bnds': 2},
                         ds0.dims)

        ds1 = ml_ds.get_dataset(1)
        self.assertIsNot(ds, ds1)
        self.assertEqual({'time': 14, 'lat': 360, 'lon': 720},
                         ds1.dims)

        ds2 = ml_ds.get_dataset(2)
        self.assertIsNot(ds, ds2)
        self.assertEqual({'time': 14, 'lat': 180, 'lon': 360},
                         ds2.dims)

        self.assertEqual([ds0, ds1, ds2], ml_ds.datasets)

        ml_ds.close()

    def test_arg_validation(self):
        ds = _get_test_dataset()
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            BaseMultiLevelDataset('test.levels')
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            BaseMultiLevelDataset(ds, tile_grid=512)
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            BaseMultiLevelDataset(ds, grid_mapping='crs84')


class ComputedMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ds = _get_test_dataset()

        ml_ds1 = BaseMultiLevelDataset(ds)

        def input_ml_dataset_getter(ds_id):
            if ds_id == "ml_ds1":
                return ml_ds1
            self.fail(f"unexpected ds_id={ds_id!r}")

        ml_ds2 = ComputedMultiLevelDataset(
            os.path.join(os.path.dirname(__file__),
                         "..", "webapi", "res", "test", "script.py"),
            "compute_dataset",
            ["ml_ds1"],
            input_ml_dataset_getter,
            input_parameters=dict(period='1W'),
            ds_id="ml_ds2"
        )
        self.assertEqual(3, ml_ds2.num_levels)
        self.assertIsInstance(ml_ds2.tile_grid, ImageTileGrid)
        self.assertEqual((180, 180), ml_ds2.tile_grid.tile_size)
        self.assertEqual(3, ml_ds2.tile_grid.num_levels)

        ds0 = ml_ds2.get_dataset(0)
        self.assertEqual({'time': 3, 'lat': 720, 'lon': 1440, 'bnds': 2}, ds0.dims)

        ds1 = ml_ds2.get_dataset(1)
        self.assertEqual({'time': 3, 'lat': 360, 'lon': 720}, ds1.dims)

        ds2 = ml_ds2.get_dataset(2)
        self.assertEqual({'time': 3, 'lat': 180, 'lon': 360}, ds2.dims)

        self.assertEqual([ds0, ds1, ds2], ml_ds2.datasets)

        ml_ds1.close()
        ml_ds2.close()


def _get_test_dataset(var_names=('noise',)):
    w = 1440
    h = 720
    p = 14

    x1 = -180.
    y1 = -90.
    x2 = +180.
    y2 = +90.
    dx = (x2 - x1) / w
    dy = (y2 - y1) / h

    data_vars = {var_name: (("time", "lat", "lon"), np.random.rand(p, h, w)) for var_name in var_names}

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

    return xr.Dataset(coords=coords, data_vars=data_vars).chunk(chunks=dict(lat=180, lon=180))


class ObjectStorageMultiLevelDatasetTest(S3Test):
    def test_s3_zarr(self):
        # if this test fails on travis with
        # E           IndexError: pop from an empty deque
        # then it most likly might be due to this issue: https://github.com/pydata/xarray/issues/4478
        # The issue has been fixed, but dask/s3fs does not have a new release including the fix yet

        self._write_test_cube()

        ml_ds_from_object_storage = open_ml_dataset_from_object_storage(
            'xcube-test-cube/cube-1-250-250.zarr',
            s3_kwargs=dict(key='test_fake_id', secret='test_fake_secret'),
            s3_client_kwargs=dict(endpoint_url=MOTO_SERVER_ENDPOINT_URL)
        )
        self.assertIsNotNone(ml_ds_from_object_storage)
        self.assertIn('conc_chl', ml_ds_from_object_storage.base_dataset.variables)
        self.assertEqual((2, 200, 400), ml_ds_from_object_storage.base_dataset.conc_chl.shape)

    @classmethod
    def _write_test_cube(cls):
        s3_kwargs = dict(key='test_fake_id', secret='test_fake_secret')
        s3_client_kwargs = dict(endpoint_url=MOTO_SERVER_ENDPOINT_URL)
        s3 = s3fs.S3FileSystem(**s3_kwargs, client_kwargs=s3_client_kwargs)
        # Create bucket 'xcube-test', so it exists before we write a test cube
        s3.mkdir('xcube-test-cube')

        # Create a test cube with just one variable "conc_chl"
        zarr_path = os.path.join(os.path.dirname(__file__), '../../examples/serve/demo/cube-1-250-250.zarr')
        dataset = xr.open_zarr(zarr_path)
        dataset = cls._make_subset(dataset)

        # Write test cube
        write_dataset(dataset,
                      'xcube-test-cube/cube-1-250-250.zarr',
                      s3_kwargs=s3_kwargs,
                      s3_client_kwargs=s3_client_kwargs)

    def test_s3_levels(self):
        self._write_test_cube_pyramid()

        s3 = s3fs.S3FileSystem(key='test_fake_id',
                               secret='test_fake_secret',
                               client_kwargs=dict(endpoint_url=MOTO_SERVER_ENDPOINT_URL))
        ml_dataset = ObjectStorageMultiLevelDataset(s3,
                                                    "xcube-test/cube-1-250-250.levels",
                                                    chunk_cache_capacity=1000 * 1000 * 1000)
        self.assertIsNotNone(ml_dataset)
        self.assertEqual(2, ml_dataset.num_levels)
        self.assertEqual((100, 100), ml_dataset.tile_grid.tile_size)
        self.assertEqual((2, 1), ml_dataset.tile_grid.get_num_tiles(0))
        self.assertEqual(800000000, ml_dataset.get_chunk_cache_capacity(0))
        self.assertEqual(200000000, ml_dataset.get_chunk_cache_capacity(1))

    @classmethod
    def _write_test_cube_pyramid(cls):
        s3_kwargs = dict(key='test_fake_id', secret='test_fake_secret')
        s3_client_kwargs = dict(endpoint_url=MOTO_SERVER_ENDPOINT_URL)
        s3 = s3fs.S3FileSystem(**s3_kwargs, client_kwargs=s3_client_kwargs)
        # Create bucket 'xcube-test', so it exists before we write a test pyramid
        s3.mkdir('xcube-test')

        # Create a test cube pyramid with just one variable "conc_chl"
        zarr_path = os.path.join(os.path.dirname(__file__), '../../examples/serve/demo/cube-1-250-250.zarr')
        base_dataset = xr.open_zarr(zarr_path)
        base_dataset = cls._make_subset(base_dataset)
        ml_dataset = BaseMultiLevelDataset(base_dataset)

        # Write test cube pyramid
        write_levels(ml_dataset,
                     'xcube-test/cube-1-250-250.levels',
                     s3_kwargs=s3_kwargs,
                     s3_client_kwargs=s3_client_kwargs)

    @classmethod
    def _make_subset(cls, base_dataset):
        base_dataset = base_dataset.drop_vars(['c2rcc_flags', 'conc_tsm', 'kd489', 'quality_flags'])
        geometry = (1.0, 51.0, 2.0, 51.5)
        base_dataset = clip_dataset_by_geometry(base_dataset, geometry=geometry)
        base_dataset = base_dataset.dropna('time', how='all')
        # somehow clip_dataset_by_geometry() does not unify chunks and encoding, so it needs to be done manually:
        for var in base_dataset.variables:
            if var not in base_dataset.coords:
                if base_dataset[var].encoding['chunks'] != (1, 100, 100):
                    base_dataset[var].encoding['chunks'] = (1, 100, 100)
                base_dataset[var] = base_dataset[var].chunk({'time': 1, 'lat': 100, 'lon': 100})
        base_dataset['lat'] = base_dataset['lat'].chunk({'lat': 100})
        base_dataset.lat.encoding['chunks'] = (100,)
        base_dataset['lon'] = base_dataset['lon'].chunk({'lon': 100})
        base_dataset.lon.encoding['chunks'] = (100,)
        base_dataset['lat_bnds'] = base_dataset['lat_bnds'].chunk((100, 2))
        base_dataset.lat_bnds.encoding['chunks'] = (100, 2)
        base_dataset['lon_bnds'] = base_dataset['lon_bnds'].chunk((100, 2))
        base_dataset.lon_bnds.encoding['chunks'] = (100, 2)
        return base_dataset
