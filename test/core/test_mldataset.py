import os
import unittest

import boto3
import moto
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from xcube.core.dsio import write_dataset
from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import CombinedMultiLevelDataset
from xcube.core.mldataset import ComputedMultiLevelDataset
from xcube.core.mldataset import ObjectStorageMultiLevelDataset
from xcube.core.mldataset import get_dataset_tile_grid
from xcube.core.mldataset import open_ml_dataset_from_object_storage
from xcube.core.mldataset import write_levels
from xcube.core.new import new_cube
from xcube.util.tilegrid import TileGrid


class CombinedMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ml_ds_1 = BaseMultiLevelDataset(_get_test_dataset(('noise_1', 'noise_2')))
        ml_ds_2 = BaseMultiLevelDataset(_get_test_dataset(('noise_3', 'noise_4')))
        ml_ds_3 = BaseMultiLevelDataset(_get_test_dataset(('noise_5', 'noise_6')))

        ml_ds = CombinedMultiLevelDataset([ml_ds_1, ml_ds_2, ml_ds_3])

        self.assertEqual(3, ml_ds.num_levels)
        self.assertEqual(TileGrid(3, 2, 1, 180, 180, (-180, -90, 180, 90), inv_y=False),
                         ml_ds.tile_grid)

        expected_var_names = {'noise_1', 'noise_2',
                              'noise_3', 'noise_4',
                              'noise_5', 'noise_6'}

        ds0 = ml_ds.get_dataset(0)
        self.assertEqual({'time': 14, 'lat': 720, 'lon': 1440, 'bnds': 2}, ds0.dims)
        self.assertEqual(expected_var_names, set(map(str, ds0.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon') for v in ds0.data_vars.values()))

        ds1 = ml_ds.get_dataset(1)
        self.assertEqual({'time': 14, 'lat': 360, 'lon': 720}, ds1.dims)
        self.assertEqual(expected_var_names, set(map(str, ds1.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon') for v in ds1.data_vars.values()))

        ds2 = ml_ds.get_dataset(2)
        self.assertEqual({'time': 14, 'lat': 180, 'lon': 360}, ds2.dims)
        self.assertEqual(expected_var_names, set(map(str, ds2.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon') for v in ds2.data_vars.values()))

        self.assertEqual([ds0, ds1, ds2], ml_ds.datasets)

        ml_ds.close()


class BaseMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ds = _get_test_dataset()

        ml_ds = BaseMultiLevelDataset(ds)

        self.assertIsInstance(ml_ds.ds_id, str)

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

        self.assertEqual([ds0, ds1, ds2], ml_ds.datasets)

        ml_ds.close()


class ComputedMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ds = _get_test_dataset()

        ml_ds1 = BaseMultiLevelDataset(ds)

        def input_ml_dataset_getter(ds_id):
            if ds_id == "ml_ds1":
                return ml_ds1
            self.fail(f"unexpected ds_id={ds_id!r}")

        ml_ds2 = ComputedMultiLevelDataset(os.path.join(os.path.dirname(__file__),
                                                        "..", "webapi", "res", "test", "script.py"),
                                           "compute_dataset",
                                           ["ml_ds1"],
                                           input_ml_dataset_getter,
                                           input_parameters=dict(period='1W'),
                                           ds_id="ml_ds2")
        self.assertEqual(3, ml_ds2.num_levels)
        self.assertEqual(TileGrid(3, 2, 1, 180, 180, (-180, -90, 180, 90), inv_y=False),
                         ml_ds2.tile_grid)

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

    return xr.Dataset(coords=coords, data_vars=data_vars)


class ObjectStorageMultiLevelDatasetTest(unittest.TestCase):
    def test_s3_zarr(self):
        with moto.mock_s3():
            self._write_test_cube()

            ml_ds_from_object_storage = open_ml_dataset_from_object_storage(
                'https://s3.amazonaws.com/xcube-test/cube-1-250-250.zarr',
                client_kwargs=dict(
                    provider_access_key_id='test_fake_id',
                    provider_secret_access_key='test_fake_secret'
                )
            )
            self.assertIsNotNone(ml_ds_from_object_storage)
            self.assertIn('conc_chl', ml_ds_from_object_storage.base_dataset.variables)
            self.assertEqual((5, 1000, 2000), ml_ds_from_object_storage.base_dataset.conc_chl.shape)

    @classmethod
    def _write_test_cube(cls):
        # Create bucket 'xcube-test', so it exists before we write a test pyramid
        s3_conn = boto3.client('s3')
        s3_conn.create_bucket(Bucket='xcube-test', ACL='public-read')

        # Create a test cube with just one variable "conc_chl"
        zarr_path = os.path.join(os.path.dirname(__file__), '../../examples/serve/demo/cube-1-250-250.zarr')
        dataset = xr.open_zarr(zarr_path)
        dataset = xr.Dataset(dict(conc_chl=dataset.conc_chl))

        # Write test cube
        write_dataset(dataset,
                      'https://s3.amazonaws.com/xcube-test/cube-1-250-250.zarr',
                      client_kwargs=dict(provider_access_key_id='test_fake_id',
                                         provider_secret_access_key='test_fake_secret'))

    def test_s3_levels(self):
        with moto.mock_s3():
            self._write_test_cube_pyramid()

            s3 = s3fs.S3FileSystem(key='test_fake_id',
                                   secret='test_fake_secret',
                                   client_kwargs=dict(endpoint_url="https://s3.amazonaws.com"))
            ml_dataset = ObjectStorageMultiLevelDataset(s3,
                                                        "xcube-test/cube-1-250-250.levels",
                                                        chunk_cache_capacity=1000 * 1000 * 1000)
            self.assertIsNotNone(ml_dataset)
            self.assertEqual(3, ml_dataset.num_levels)
            self.assertEqual((250, 250), ml_dataset.tile_grid.tile_size)
            self.assertEqual(2, ml_dataset.tile_grid.num_level_zero_tiles_x)
            self.assertEqual(1, ml_dataset.tile_grid.num_level_zero_tiles_y)
            self.assertEqual(761904762, ml_dataset.get_chunk_cache_capacity(0))
            self.assertEqual(190476190, ml_dataset.get_chunk_cache_capacity(1))
            self.assertEqual(47619048, ml_dataset.get_chunk_cache_capacity(2))

    @classmethod
    def _write_test_cube_pyramid(cls):
        # Create bucket 'xcube-test', so it exists before we write a test pyramid
        s3_conn = boto3.client('s3')
        s3_conn.create_bucket(Bucket='xcube-test', ACL='public-read')

        # Create a test cube pyramid with just one variable "conc_chl"
        zarr_path = os.path.join(os.path.dirname(__file__), '../../examples/serve/demo/cube-1-250-250.zarr')
        base_dataset = xr.open_zarr(zarr_path)
        base_dataset = xr.Dataset(dict(conc_chl=base_dataset.conc_chl))
        ml_dataset = BaseMultiLevelDataset(base_dataset)

        # Write test cube pyramid
        write_levels(ml_dataset,
                     'https://s3.amazonaws.com/xcube-test/cube-1-250-250.levels',
                     client_kwargs=dict(provider_access_key_id='test_fake_id',
                                        provider_secret_access_key='test_fake_secret'))


class GetDatasetTileGridTest(unittest.TestCase):

    def setUp(self) -> None:
        self.cube = new_cube(width=2000, height=1000,
                             x_start=0, x_res=0.0025,
                             y_start=50, y_res=0.0025,
                             inverse_y=True,
                             variables={'a': 0.5})

    def test_no_chunks(self):
        tile_grid = get_dataset_tile_grid(self.cube)
        self.assertEqual(3, tile_grid.num_levels)
        self.assertEqual((250, 250), tile_grid.tile_size)
        self.assertEqual(2, tile_grid.num_level_zero_tiles_x)
        self.assertEqual(1, tile_grid.num_level_zero_tiles_y)
        self.assertEqual((0, 50, 5, 52.5), tile_grid.geo_extent)
        self.assertEqual((500, 250), tile_grid.size(0))
        self.assertEqual((1000, 500), tile_grid.size(1))
        self.assertEqual((2000, 1000), tile_grid.size(2))

    def test_no_chunks_num_levels(self):
        tile_grid = get_dataset_tile_grid(self.cube, num_levels=3)
        self.assertEqual(3, tile_grid.num_levels)
        self.assertEqual((250, 250), tile_grid.tile_size)
        self.assertEqual(2, tile_grid.num_level_zero_tiles_x)
        self.assertEqual(1, tile_grid.num_level_zero_tiles_y)
        self.assertEqual((0, 50, 5, 52.5), tile_grid.geo_extent)
        self.assertEqual((500, 250), tile_grid.size(0))
        self.assertEqual((1000, 500), tile_grid.size(1))
        self.assertEqual((2000, 1000), tile_grid.size(2))

    def test_chunks(self):
        tile_grid = get_dataset_tile_grid(self.cube.chunk(dict(time=1, lat=250, lon=250)))
        self.assertEqual(3, tile_grid.num_levels)
        self.assertEqual((250, 250), tile_grid.tile_size)
        self.assertEqual(2, tile_grid.num_level_zero_tiles_x)
        self.assertEqual(1, tile_grid.num_level_zero_tiles_y)
        self.assertEqual((0, 50, 5, 52.5), tile_grid.geo_extent)
        self.assertEqual((500, 250), tile_grid.size(0))
        self.assertEqual((1000, 500), tile_grid.size(1))
        self.assertEqual((2000, 1000), tile_grid.size(2))

        tile_grid = get_dataset_tile_grid(self.cube.chunk(dict(time=1, lat=256, lon=512)))
        self.assertEqual(3, tile_grid.num_levels)
        self.assertEqual((500, 250), tile_grid.tile_size)
        self.assertEqual(1, tile_grid.num_level_zero_tiles_x)
        self.assertEqual(1, tile_grid.num_level_zero_tiles_y)
        self.assertEqual((0, 50, 5, 52.5), tile_grid.geo_extent)
        self.assertEqual((500, 250), tile_grid.size(0))
        self.assertEqual((1000, 500), tile_grid.size(1))
        self.assertEqual((2000, 1000), tile_grid.size(2))

    def test_chunks_num_levels(self):
        tile_grid = get_dataset_tile_grid(self.cube.chunk(dict(time=1, lat=250, lon=250)), num_levels=3)
        self.assertEqual(3, tile_grid.num_levels)
        self.assertEqual((250, 250), tile_grid.tile_size)
        self.assertEqual(2, tile_grid.num_level_zero_tiles_x)
        self.assertEqual(1, tile_grid.num_level_zero_tiles_y)
        self.assertEqual((0, 50, 5, 52.5), tile_grid.geo_extent)
        self.assertEqual((500, 250), tile_grid.size(0))
        self.assertEqual((1000, 500), tile_grid.size(1))
        self.assertEqual((2000, 1000), tile_grid.size(2))

        tile_grid = get_dataset_tile_grid(self.cube.chunk(dict(time=1, lat=256, lon=256)), num_levels=4)
        self.assertEqual(4, tile_grid.num_levels)
        self.assertEqual((256, 256), tile_grid.tile_size)
        self.assertEqual(1, tile_grid.num_level_zero_tiles_x)
        self.assertEqual(1, tile_grid.num_level_zero_tiles_y)
        self.assertEqual((0, 50, 5, 52.5), tile_grid.geo_extent)
        self.assertEqual((256, 256), tile_grid.size(0))
        self.assertEqual((512, 512), tile_grid.size(1))
        self.assertEqual((1024, 1024), tile_grid.size(2))
        self.assertEqual((2048, 2048), tile_grid.size(3))
