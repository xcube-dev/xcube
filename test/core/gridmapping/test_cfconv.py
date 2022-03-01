import shutil
import unittest
from typing import Set

import fsspec
import numpy as np
import pyproj
import xarray as xr

from test.s3test import MOTO_SERVER_ENDPOINT_URL
from test.s3test import S3Test
from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.cfconv import GridCoords
from xcube.core.gridmapping.cfconv import GridMappingProxy
from xcube.core.gridmapping.cfconv import add_spatial_ref
from xcube.core.gridmapping.cfconv import get_dataset_grid_mapping_proxies
from xcube.core.new import new_cube
from xcube.core.store.fs.registry import new_fs_data_store

CRS_WGS84 = pyproj.crs.CRS(4326)
CRS_CRS84 = pyproj.crs.CRS.from_string("urn:ogc:def:crs:OGC:1.3:CRS84")
CRS_UTM_33N = pyproj.crs.CRS(32633)

CRS_ROTATED_POLE = pyproj.crs.CRS.from_cf(
    dict(grid_mapping_name="rotated_latitude_longitude",
         grid_north_pole_latitude=32.5,
         grid_north_pole_longitude=170.))


class GetDatasetGridMappingsTest(unittest.TestCase):
    def test_no_crs_lon_lat_common_names(self):
        dataset = xr.Dataset(coords=dict(lon=xr.DataArray(np.linspace(10, 12, 11), dims='lon'),
                                         lat=xr.DataArray(np.linspace(50, 52, 11), dims='lat')))
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn(None, grid_mappings)
        grid_mapping = grid_mappings.get(None)
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_WGS84, grid_mapping.crs)
        self.assertEqual('latitude_longitude', grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('lon', grid_mapping.coords.x.name)
        self.assertEqual('lat', grid_mapping.coords.y.name)

    def test_no_crs_lon_lat_standard_names(self):
        dataset = xr.Dataset(coords=dict(weird_x=xr.DataArray(np.linspace(10, 12, 11), dims='i',
                                                              attrs=dict(standard_name='longitude')),
                                         weird_y=xr.DataArray(np.linspace(50, 52, 11), dims='j',
                                                              attrs=dict(standard_name='latitude'))))
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn(None, grid_mappings)
        grid_mapping = grid_mappings.get(None)
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_WGS84, grid_mapping.crs)
        self.assertEqual('latitude_longitude', grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('weird_x', grid_mapping.coords.x.name)
        self.assertEqual('weird_y', grid_mapping.coords.y.name)

    def test_crs_x_y_with_common_names(self):
        dataset = xr.Dataset(dict(crs=xr.DataArray(0, attrs=CRS_UTM_33N.to_cf())),
                             coords=dict(x=xr.DataArray(np.linspace(1000, 12000, 11), dims='x'),
                                         y=xr.DataArray(np.linspace(5000, 52000, 11), dims='y')))
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn('crs', grid_mappings)
        grid_mapping = grid_mappings.get('crs')
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_UTM_33N, grid_mapping.crs)
        self.assertEqual('transverse_mercator', grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('x', grid_mapping.coords.x.name)
        self.assertEqual('y', grid_mapping.coords.y.name)

    def test_crs_x_y_with_standard_names(self):
        dataset = xr.Dataset(dict(crs=xr.DataArray(0, attrs=CRS_UTM_33N.to_cf())),
                             coords=dict(myx=xr.DataArray(np.linspace(1000, 12000, 11), dims='x',
                                                          attrs=dict(standard_name='projection_x_coordinate')),
                                         myy=xr.DataArray(np.linspace(5000, 52000, 11), dims='y',
                                                          attrs=dict(standard_name='projection_y_coordinate'))))
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn('crs', grid_mappings)
        grid_mapping = grid_mappings.get('crs')
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual(CRS_UTM_33N, grid_mapping.crs)
        self.assertEqual('transverse_mercator', grid_mapping.name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('myx', grid_mapping.coords.x.name)
        self.assertEqual('myy', grid_mapping.coords.y.name)

    def test_rotated_pole_with_common_names(self):
        dataset = xr.Dataset(dict(rotated_pole=xr.DataArray(0, attrs=CRS_ROTATED_POLE.to_cf())),
                             coords=dict(rlon=xr.DataArray(np.linspace(-180, 180, 11), dims='rlon'),
                                         rlat=xr.DataArray(np.linspace(0, 90, 11), dims='rlat')))
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn('rotated_pole', grid_mappings)
        grid_mapping = grid_mappings.get('rotated_pole')
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual('Geographic 2D CRS', grid_mapping.crs.type_name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('rlon', grid_mapping.coords.x.name)
        self.assertEqual('rlat', grid_mapping.coords.y.name)

    def test_rotated_pole_with_standard_names(self):
        dataset = xr.Dataset(dict(rotated_pole=xr.DataArray(0, attrs=CRS_ROTATED_POLE.to_cf())),
                             coords=dict(u=xr.DataArray(np.linspace(-180, 180, 11), dims='u',
                                                        attrs=dict(standard_name='grid_longitude')),
                                         v=xr.DataArray(np.linspace(0, 90, 11), dims='v',
                                                        attrs=dict(standard_name='grid_latitude'))))
        grid_mappings = get_dataset_grid_mapping_proxies(dataset)
        self.assertEqual(1, len(grid_mappings))
        self.assertIn('rotated_pole', grid_mappings)
        grid_mapping = grid_mappings.get('rotated_pole')
        self.assertIsInstance(grid_mapping, GridMappingProxy)
        self.assertEqual('Geographic 2D CRS', grid_mapping.crs.type_name)
        self.assertIsInstance(grid_mapping.coords, GridCoords)
        self.assertIsInstance(grid_mapping.coords.x, xr.DataArray)
        self.assertIsInstance(grid_mapping.coords.y, xr.DataArray)
        self.assertEqual('u', grid_mapping.coords.x.name)
        self.assertEqual('v', grid_mapping.coords.y.name)


class XarrayDecodeCfTest(unittest.TestCase):
    """Find out how xarray treats 1D and 2D coordinate variables when decode_cf=True or =False"""

    def test_cf_1d_coords(self):
        self._write_coords(*self._gen_1d())
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=True)
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=False)

    def test_cf_1d_data_vars(self):
        self._write_data_vars(*self._gen_1d())
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=True)
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=False)

    def test_cf_2d_coords(self):
        self._write_coords(*self._gen_2d())
        self.assertVarNames({'noise', 'crs'}, {'lon', 'lat'}, decode_cf=True)
        self.assertVarNames({'noise', 'crs', 'lon', 'lat'}, set(), decode_cf=False)

    def test_cf_2d_data_vars(self):
        self._write_data_vars(*self._gen_2d())
        self.assertVarNames({'noise', 'crs', 'lon', 'lat'}, set(), decode_cf=True)
        self.assertVarNames({'noise', 'crs', 'lon', 'lat'}, set(), decode_cf=False)

    def assertVarNames(self, exp_data_vars: Set[str], exp_coords: Set[str], decode_cf: bool):
        ds1 = xr.open_zarr('noise.zarr', decode_cf=decode_cf)
        self.assertEqual(exp_coords, set(ds1.coords))
        self.assertEqual(exp_data_vars, set(ds1.data_vars))

    @classmethod
    def _write_coords(cls, noise, crs, lon, lat):
        dataset = xr.Dataset(dict(noise=noise, crs=crs), coords=dict(lon=lon, lat=lat))
        dataset.to_zarr('noise.zarr', mode='w')

    @classmethod
    def _write_data_vars(cls, noise, crs, lon, lat):
        dataset = xr.Dataset(dict(noise=noise, crs=crs, lon=lon, lat=lat))
        dataset.to_zarr('noise.zarr', mode='w')

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('noise.zarr')

    def _gen_1d(self):
        noise = xr.DataArray(np.random.random((11, 11)), dims=('lat', 'lon'))
        crs = xr.DataArray(0, attrs=CRS_CRS84.to_cf())
        lon = xr.DataArray(np.linspace(10, 12, 11), dims='lon')
        lat = xr.DataArray(np.linspace(50, 52, 11), dims='lat')
        noise.attrs['grid_mapping'] = 'crs'
        lon.attrs['standard_name'] = 'longitude'
        lat.attrs['standard_name'] = 'latitude'
        self.assertEqual(('lon',), lon.dims)
        self.assertEqual(('lat',), lat.dims)
        return noise, crs, lon, lat

    def _gen_2d(self):
        noise = xr.DataArray(np.random.random((11, 11)), dims=('y', 'x'))
        crs = xr.DataArray(0, attrs=CRS_CRS84.to_cf())
        lon = xr.DataArray(np.linspace(10, 12, 11), dims='x')
        lat = xr.DataArray(np.linspace(50, 52, 11), dims='y')
        lat, lon = xr.broadcast(lat, lon)
        noise.attrs['grid_mapping'] = 'crs'
        lon.attrs['standard_name'] = 'longitude'
        lat.attrs['standard_name'] = 'latitude'
        self.assertEqual(('y', 'x'), lon.dims)
        self.assertEqual(('y', 'x'), lat.dims)
        return noise, crs, lon, lat


class AddSpatialRefTest(S3Test):

    def test_add_spatial_ref(self):
        self.assert_add_spatial_ref_ok(None, None)
        self.assert_add_spatial_ref_ok(None, ('cx', 'cy'))
        self.assert_add_spatial_ref_ok('crs', None)
        self.assert_add_spatial_ref_ok('crs', ('cx', 'cy'))

    def assert_add_spatial_ref_ok(self, crs_var_name, xy_dim_names):

        root = 'eurodatacube-test/xcube-eea'
        data_id = 'test.zarr'
        crs = pyproj.CRS.from_string("EPSG:3035")

        if xy_dim_names:
            x_name, y_name = xy_dim_names
        else:
            x_name, y_name = 'x', 'y'

        cube = new_cube(x_name=x_name,
                        y_name=y_name,
                        x_start=0,
                        y_start=0,
                        x_res=10,
                        y_res=10,
                        x_units='metres',
                        y_units='metres',
                        drop_bounds=True,
                        width=100,
                        height=100,
                        variables=dict(A=1.3, B=8.3))

        storage_options = dict(
            anon=False,
            client_kwargs=dict(
                endpoint_url=MOTO_SERVER_ENDPOINT_URL,
            )
        )

        fs: fsspec.AbstractFileSystem = fsspec.filesystem('s3',
                                                          **storage_options)
        if fs.isdir(root):
            fs.rm(root, recursive=True)
        fs.mkdirs(root, exist_ok=True)

        data_store = new_fs_data_store('s3',
                                       root=root,
                                       storage_options=storage_options)

        data_store.write_data(cube, data_id=data_id)
        cube = data_store.open_data(data_id)
        self.assertEqual({'A', 'B', 'time', x_name, y_name},
                         set(cube.variables))

        with self.assertRaises(ValueError) as cm:
            GridMapping.from_dataset(cube)
        self.assertEqual(
            ('cannot find any grid mapping in dataset',),
            cm.exception.args
        )

        path = f"{root}/{data_id}"
        group_store = fs.get_mapper(path, create=True)

        expected_crs_var_name = crs_var_name or 'spatial_ref'

        self.assertTrue(fs.exists(path))
        self.assertFalse(fs.exists(f"{path}/{expected_crs_var_name}"))
        self.assertFalse(fs.exists(f"{path}/{expected_crs_var_name}/.zarray"))
        self.assertFalse(fs.exists(f"{path}/{expected_crs_var_name}/.zattrs"))

        kwargs = {}
        if crs_var_name is not None:
            kwargs.update(crs_var_name=crs_var_name)
        if xy_dim_names is not None:
            kwargs.update(xy_dim_names=xy_dim_names)
        add_spatial_ref(group_store, crs, **kwargs)

        self.assertTrue(fs.exists(f"{path}/{expected_crs_var_name}"))
        self.assertTrue(fs.exists(f"{path}/{expected_crs_var_name}/.zarray"))
        self.assertTrue(fs.exists(f"{path}/{expected_crs_var_name}/.zattrs"))

        cube = data_store.open_data(data_id)
        self.assertEqual({'A', 'B', 'time',
                          x_name, y_name, expected_crs_var_name},
                         set(cube.variables))
        self.assertEqual(expected_crs_var_name,
                         cube.A.attrs.get('grid_mapping'))
        self.assertEqual(expected_crs_var_name,
                         cube.B.attrs.get('grid_mapping'))

        gm = GridMapping.from_dataset(cube)
        self.assertIn("LAEA Europe", gm.crs.srs)
