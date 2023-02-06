import unittest

import dask.array as da
import geopandas as gpd
import numpy as np
import shapely.geometry
import shapely.wkt
import xarray as xr

from xcube.core.chunk import chunk_dataset
from xcube.core.geom import clip_dataset_by_geometry
from xcube.core.geom import convert_geometry
from xcube.core.geom import get_dataset_bounds
from xcube.core.geom import get_dataset_geometry
from xcube.core.geom import is_dataset_y_axis_inverted
from xcube.core.geom import is_lon_lat_dataset
from xcube.core.geom import mask_dataset_by_geometry
from xcube.core.geom import normalize_geometry
from xcube.core.geom import rasterize_features
from xcube.core.new import new_cube
from xcube.util.types import normalize_scalar_or_pair

nan = np.nan


class RasterizeFeaturesIntoDataset(unittest.TestCase):
    def test_rasterize_geo_data_frame_lonlat(self):
        self._test_rasterize_features(self.get_geo_data_frame_features(),
                                      'lon', 'lat')

    def test_rasterize_geo_data_frame_lonlat_chunked(self):
        self._test_rasterize_features(self.get_geo_data_frame_features(),
                                      'lon', 'lat', with_var=True)

    def test_rasterize_geo_data_frame_lonlat_fix_chunks(self):
        self._test_rasterize_features(self.get_geo_data_frame_features(),
                                      'lon', 'lat', tile_size=4)

    def test_rasterize_geo_data_frame_xy(self):
        self._test_rasterize_features(self.get_geo_data_frame_features(),
                                      'x', 'y')

    def test_rasterize_geo_data_frame_xy_chunked(self):
        self._test_rasterize_features(self.get_geo_data_frame_features(),
                                      'x', 'y', with_var=True)

    def test_rasterize_geo_data_frame_xy_fix_chunks(self):
        self._test_rasterize_features(self.get_geo_data_frame_features(),
                                      'x', 'y', tile_size=(3, 4))

    def test_rasterize_geo_json_lonlat(self):
        self._test_rasterize_features(self.get_geo_json_features(),
                                      'lon', 'lat')

    def test_rasterize_geo_json_lonlat_chunked(self):
        self._test_rasterize_features(self.get_geo_json_features(),
                                      'lon', 'lat', with_var=True)

    def test_rasterize_geo_json_xy(self):
        self._test_rasterize_features(self.get_geo_json_features(),
                                      'x', 'y')

    def test_rasterize_geo_json_xy_chunked(self):
        self._test_rasterize_features(self.get_geo_json_features(),
                                      'x', 'y', with_var=True)

    def test_rasterize_geo_json_xy_chunked_inverse_y(self):
        self._test_rasterize_features(self.get_geo_json_features(),
                                      'x', 'y',
                                      with_var=True,
                                      inverse_y=True)

    def test_rasterize_geo_json_xy_fixed_chunks_inverse_y(self):
        self._test_rasterize_features(self.get_geo_json_features(),
                                      'x', 'y',
                                      tile_size=4,
                                      inverse_y=True)

    def test_rasterize_invalid_feature(self):
        features = self.get_geo_json_features()
        with self.assertRaises(ValueError) as cm:
            rasterize_features(new_cube(),
                               features,
                               ['missing'])
        self.assertEqual("feature property 'missing' not found",
                         f'{cm.exception}')

    def _test_rasterize_features(self,
                                 features,
                                 x_name, y_name,
                                 with_var=False,
                                 tile_size=None,
                                 inverse_y=False):
        width = 10
        height = 10
        dataset = new_cube(
            width=width, height=height,
            x_name=x_name, y_name=y_name,
            x_res=10,
            x_start=-50, y_start=-50,
            variables=dict(d=12.5) if with_var else None,
            inverse_y=inverse_y
        )
        if with_var:
            dataset['d'] = dataset['d'].chunk({x_name: 4,
                                               y_name: 3,
                                               'time': 1})

        dataset = rasterize_features(dataset,
                                     features,
                                     ['a', 'b', 'c'],
                                     var_props=dict(
                                         b=dict(name='b',
                                                dtype=np.float32,
                                                fill_value=np.nan,
                                                attrs=dict(units='meters')),
                                         c=dict(name='c2',
                                                dtype=np.uint8,
                                                fill_value=0,
                                                converter=int)
                                     ),
                                     tile_size=tile_size,
                                     in_place=False)

        self.assertIsNotNone(dataset)
        if with_var:
            cy, cx = (3, 3, 3, 1), (4, 4, 2)
        elif tile_size:
            tw, th = normalize_scalar_or_pair(tile_size)
            cy, cx = da.core.normalize_chunks((th, tw), shape=(height, width))
        else:
            cy, cx = (10,), (10,)
        if not inverse_y:
            cy = tuple(reversed(cy))
        self.assertEqual({x_name: cx, y_name: cy}, dataset.chunks)

        self.assertIn(x_name, dataset.coords)
        self.assertIn(y_name, dataset.coords)
        self.assertIn('time', dataset.coords)
        self.assertIn('a', dataset)
        self.assertIn('b', dataset)
        self.assertIn('c2', dataset)
        self.assertEquals((10, 10), dataset.a.shape)
        self.assertEquals((10, 10), dataset.b.shape)
        self.assertEquals((10, 10), dataset.c2.shape)

        # Assert in-memory (decoding) data types are correct.
        self.assertEquals(np.float64, dataset.a.dtype)
        self.assertEquals(np.float64, dataset.b.dtype)
        self.assertEquals(np.float64, dataset.c2.dtype)

        # Assert external representation (encoding) information
        # is correctly set up.
        # See also test.core.test_xarray.XarrayEncodingTest
        self.assertIs(nan, dataset.a.encoding.get('_FillValue'))
        self.assertIs(nan, dataset.b.encoding.get('_FillValue'))
        self.assertEqual(0, dataset.c2.encoding.get('_FillValue'))
        self.assertEqual(np.dtype('float64'), dataset.a.encoding.get('dtype'))
        self.assertEqual(np.dtype('float32'), dataset.b.encoding.get('dtype'))
        self.assertEqual(np.dtype('uint8'), dataset.c2.encoding.get('dtype'))

        # Other metadata
        self.assertEquals({}, dataset.a.attrs)
        self.assertEquals({'units': 'meters'}, dataset.b.attrs)
        self.assertEquals({}, dataset.c2.attrs)
        self.assertEquals((y_name, x_name), dataset.a.dims)
        self.assertEquals((y_name, x_name), dataset.b.dims)
        self.assertEquals((y_name, x_name), dataset.c2.dims)

        # Assert actual data is correct

        actual_a_values = dataset.a.values
        expected_a_values = np.array(
            [[0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
             [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
             [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
             [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
             [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
             [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
             [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
             [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
             [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
             [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7]]
        )
        if inverse_y:
            expected_a_values = expected_a_values[::-1, :]
        np.testing.assert_almost_equal(expected_a_values,
                                       actual_a_values)
        actual_b_values = dataset.b.values
        expected_b_values = np.array(
            [[2.2, 2.2, 2.2, 2.2, 2.2, nan, nan, 2.4, 2.4, 2.4],
             [2.2, 2.2, 2.2, 2.2, 2.2, nan, nan, 2.4, 2.4, 2.4],
             [2.2, 2.2, 2.2, 2.2, 2.2, nan, nan, 2.4, 2.4, 2.4],
             [2.2, 2.2, 2.2, 2.2, 2.2, nan, nan, 2.4, 2.4, 2.4],
             [2.2, 2.2, 2.2, 2.2, 2.2, nan, nan, 2.4, 2.4, 2.4],
             [2.1, 2.1, 2.1, 2.1, 2.1, nan, nan, 2.3, 2.3, 2.3],
             [2.1, 2.1, 2.1, 2.1, 2.1, nan, nan, 2.3, 2.3, 2.3],
             [2.1, 2.1, 2.1, 2.1, 2.1, nan, nan, 2.3, 2.3, 2.3],
             [2.1, 2.1, 2.1, 2.1, 2.1, nan, nan, 2.3, 2.3, 2.3],
             [2.1, 2.1, 2.1, 2.1, 2.1, nan, nan, 2.3, 2.3, 2.3]]
        )
        if inverse_y:
            expected_b_values = expected_b_values[::-1, :]
        np.testing.assert_almost_equal(expected_b_values,
                                       actual_b_values)
        actual_c_values = dataset.c2.values
        expected_c_values = np.array(
            [[8, 8, 8, 8, 8, nan, nan, 6, 6, 6],
             [8, 8, 8, 8, 8, nan, nan, 6, 6, 6],
             [8, 8, 8, 8, 8, nan, nan, 6, 6, 6],
             [8, 8, 8, 8, 8, nan, nan, 6, 6, 6],
             [8, 8, 8, 8, 8, nan, nan, 6, 6, 6],
             [9, 9, 9, 9, 9, nan, nan, 7, 7, 7],
             [9, 9, 9, 9, 9, nan, nan, 7, 7, 7],
             [9, 9, 9, 9, 9, nan, nan, 7, 7, 7],
             [9, 9, 9, 9, 9, nan, nan, 7, 7, 7],
             [9, 9, 9, 9, 9, nan, nan, 7, 7, 7]]
        )
        if inverse_y:
            expected_c_values = expected_c_values[::-1, :]
        np.testing.assert_almost_equal(expected_c_values,
                                       actual_c_values)

    def get_geo_data_frame_features(self):
        features = self.get_geo_json_features()
        return gpd.GeoDataFrame.from_features(features)

    @staticmethod
    def get_geo_json_features():
        feature1 = dict(
            type='Feature',
            geometry=dict(type='Polygon',
                          coordinates=[
                              [(-180, 0), (-1, 0), (-1, 90), (-180, 90),
                               (-180, 0)]]),
            properties=dict(a=0.5, b=2.1, c=9)
        )
        feature2 = dict(
            type='Feature',
            geometry=dict(type='Polygon',
                          coordinates=[
                              [(-180, -90), (-1, -90), (-1, 0), (-180, 0),
                               (-180, -90)]]),
            properties=dict(a=0.6, b=2.2, c=8)
        )
        feature3 = dict(
            type='Feature',
            geometry=dict(type='Polygon',
                          coordinates=[
                              [(20, 0), (180, 0), (180, 90), (20, 90),
                               (20, 0)]]),
            properties=dict(a=0.7, b=2.3, c=7)
        )
        feature4 = dict(
            type='Feature',
            geometry=dict(type='Polygon',
                          coordinates=[
                              [(20, -90), (180, -90), (180, 0), (20, 0),
                               (20, -90)]]),
            properties=dict(a=0.8, b=2.4, c=6)
        )
        return [feature1, feature2, feature3, feature4]


class DatasetGeometryTest(unittest.TestCase):

    def setUp(self) -> None:
        width = 16
        height = 8
        spatial_res = 360 / width
        lon_min = -2 * spatial_res + 0.5 * spatial_res
        lat_min = -2 * spatial_res + 0.5 * spatial_res
        lon_max = lon_min + 6 * spatial_res
        lat_max = lat_min + 3 * spatial_res

        self.triangle = shapely.geometry.Polygon(
            ((lon_min, lat_min),
             (lon_max, lat_min),
             (0.5 * (lon_max + lon_min), lat_max),
             (lon_min, lat_min))
        )

        self.cube = new_cube(width=width,
                             height=height,
                             x_res=spatial_res,
                             drop_bounds=True,
                             variables=dict(temp=273.9, precip=0.9))

    def test_clip_dataset_by_geometry(self):
        cube = clip_dataset_by_geometry(self.cube, self.triangle)
        self._assert_clipped_dataset_has_basic_props(cube)
        cube = clip_dataset_by_geometry(self.cube, self.triangle,
                                        save_geometry_wkt=True)
        self._assert_clipped_dataset_has_basic_props(cube)
        self._assert_saved_geometry_wkt_is_fine(cube, 'geometry_wkt')
        cube = clip_dataset_by_geometry(self.cube, self.triangle,
                                        save_geometry_wkt='intersect_geom')
        self._assert_saved_geometry_wkt_is_fine(cube, 'intersect_geom')

    def test_mask_dataset_by_geometry(self):
        cube = mask_dataset_by_geometry(self.cube, self.triangle)
        self._assert_clipped_dataset_has_basic_props(cube)
        cube = mask_dataset_by_geometry(self.cube, self.triangle,
                                        save_geometry_wkt=True)
        self._assert_saved_geometry_wkt_is_fine(cube, 'geometry_wkt')
        cube = mask_dataset_by_geometry(self.cube, self.triangle,
                                        save_geometry_wkt='intersect_geom')
        self._assert_saved_geometry_wkt_is_fine(cube, 'intersect_geom')

    def test_mask_dataset_by_geometry_excluded_vars(self):
        cube = mask_dataset_by_geometry(self.cube, self.triangle,
                                        excluded_vars='precip')
        self._assert_clipped_dataset_has_basic_props(cube)

    def test_mask_dataset_by_geometry_store_mask(self):
        cube = mask_dataset_by_geometry(self.cube, self.triangle,
                                        save_geometry_mask='geom_mask')
        self._assert_clipped_dataset_has_basic_props(cube)
        self._assert_dataset_mask_is_fine(cube, 'geom_mask')

    def test_clip_dataset_for_chunked_input(self):
        cube = chunk_dataset(self.cube,
                             chunk_sizes=dict(time=1, lat=90, lon=90))
        cube = clip_dataset_by_geometry(cube, self.triangle)
        self._assert_clipped_dataset_has_basic_props(cube)
        self.assertEqual(((1, 1, 1, 1, 1), (4,), (7,)), cube.temp.chunks)
        self.assertEqual(((1, 1, 1, 1, 1), (4,), (7,)), cube.precip.chunks)

    def test_clip_dataset_inverse_y(self):
        lat_inv = self.cube.lat[::-1]
        cube = self.cube.assign_coords(lat=lat_inv)
        cube = clip_dataset_by_geometry(cube, self.triangle)
        self._assert_clipped_dataset_has_basic_props(cube)

    def test_mask_dataset_for_chunked_input(self):
        cube = chunk_dataset(self.cube,
                             chunk_sizes=dict(time=1, lat=90, lon=90))
        cube = mask_dataset_by_geometry(cube, self.triangle)
        self._assert_clipped_dataset_has_basic_props(cube)
        self.assertEqual(((1, 1, 1, 1, 1), (4,), (7,)), cube.temp.chunks)
        self.assertEqual(((1, 1, 1, 1, 1), (4,), (7,)), cube.precip.chunks)

    def _assert_clipped_dataset_has_basic_props(self, dataset):
        self.assertEqual({'time': 5, 'lat': 4, 'lon': 7}, dataset.dims)
        self.assertIn('temp', dataset)
        self.assertIn('precip', dataset)
        temp = dataset['temp']
        precip = dataset['precip']
        self.assertEqual(('time', 'lat', 'lon'), temp.dims)
        self.assertEqual((5, 4, 7), temp.shape)
        self.assertEqual(('time', 'lat', 'lon'), precip.dims)
        self.assertEqual((5, 4, 7), precip.shape)

        self.assertIn('geospatial_lon_min', dataset.attrs)
        self.assertIn('geospatial_lon_max', dataset.attrs)
        self.assertIn('geospatial_lon_units', dataset.attrs)
        self.assertIn('geospatial_lon_resolution', dataset.attrs)
        self.assertIn('geospatial_lat_min', dataset.attrs)
        self.assertIn('geospatial_lat_max', dataset.attrs)
        self.assertIn('geospatial_lat_units', dataset.attrs)
        self.assertIn('geospatial_lat_resolution', dataset.attrs)
        self.assertIn('time_coverage_start', dataset.attrs)
        self.assertIn('time_coverage_end', dataset.attrs)
        self.assertIn('date_modified', dataset.attrs)

    def _assert_dataset_mask_is_fine(self, dataset, mask_var_name):
        self.assertIn(mask_var_name, dataset)
        geom_mask = dataset[mask_var_name]
        self.assertEqual(('lat', 'lon'), geom_mask.dims)
        self.assertEqual((4, 7), geom_mask.shape)
        actual_mask_values = geom_mask.values
        expected_mask_values = np.array(
            [
                [0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool
        )
        np.testing.assert_array_almost_equal(
            actual_mask_values,
            expected_mask_values
        )

    def _assert_saved_geometry_wkt_is_fine(self, dataset, geometry_wkt_name):
        self.assertIn(geometry_wkt_name, dataset.attrs)
        actual = shapely.wkt.loads(dataset.attrs[geometry_wkt_name])
        expected = shapely.wkt.loads(
            'POLYGON ((-33.75 -33.75, 33.75 33.75,'
            ' 101.25 -33.75, -33.75 -33.75))'
        )
        self.assertTrue(actual.difference(expected).is_empty)


class GetDatasetGeometryTest(unittest.TestCase):

    def test_nominal(self):
        ds1, ds2 = _get_nominal_datasets()
        bounds = get_dataset_geometry(ds1)
        self.assertEqual(
            shapely.geometry.box(-25.0, -15.0, 15.0, 15.0), bounds
        )
        bounds = get_dataset_geometry(ds2)
        self.assertEqual(
            shapely.geometry.box(-25.0, -15.0, 15.0, 15.0), bounds
        )

    def test_inv_y(self):
        ds1, ds2 = _get_inv_y_datasets()
        bounds = get_dataset_geometry(ds1)
        self.assertEqual(
            shapely.geometry.box(-25.0, -15.0, 15.0, 15.0), bounds
        )
        bounds = get_dataset_geometry(ds2)
        self.assertEqual(
            shapely.geometry.box(-25.0, -15.0, 15.0, 15.0), bounds
        )

    def test_antimeridian(self):
        ds1, ds2 = _get_antimeridian_datasets()
        bounds = get_dataset_geometry(ds1)
        self.assertEqual(shapely.geometry.MultiPolygon(
            polygons=[
                shapely.geometry.box(165.0, -15.0, 180.0, 15.0),
                shapely.geometry.box(-180.0, -15.0, -155.0, 15.0)
            ]),
            bounds)
        bounds = get_dataset_geometry(ds2)
        self.assertEqual(shapely.geometry.MultiPolygon(
            polygons=[
                shapely.geometry.box(165.0, -15.0, 180.0, 15.0),
                shapely.geometry.box(-180.0, -15.0, -155.0, 15.0)
            ]),
            bounds)


class GetDatasetBoundsTest(unittest.TestCase):
    def test_nominal(self):
        ds1, ds2 = _get_nominal_datasets()
        bounds = get_dataset_bounds(ds1)
        self.assertEqual((-25.0, -15.0, 15.0, 15.0), bounds)
        bounds = get_dataset_bounds(ds2)
        self.assertEqual((-25.0, -15.0, 15.0, 15.0), bounds)

    def test_longitude_latitude(self):
        ds1, ds2 = _get_nominal_datasets()
        ds1 = ds1.rename(dict(lon='longitude', lat='latitude'))
        ds2 = ds2.rename(dict(lon='longitude', lat='latitude'))

        bounds = get_dataset_bounds(ds1)
        self.assertEqual((-25.0, -15.0, 15.0, 15.0), bounds)
        bounds = get_dataset_bounds(ds2)
        self.assertEqual((-25.0, -15.0, 15.0, 15.0), bounds)

    def test_inv_y(self):
        ds1, ds2 = _get_inv_y_datasets()
        bounds = get_dataset_bounds(ds1)
        self.assertEqual((-25.0, -15.0, 15.0, 15.0), bounds)
        bounds = get_dataset_bounds(ds2)
        self.assertEqual((-25.0, -15.0, 15.0, 15.0), bounds)

    def test_inv_y_wrong_order_bounds(self):
        ds1, ds2 = _get_inv_y_wrong_order_bounds_datasets()
        bounds = get_dataset_bounds(ds1)
        self.assertEqual((-25.0, -15.0, 15.0, 15.0), bounds)
        bounds = get_dataset_bounds(ds2)
        self.assertEqual((-25.0, -15.0, 15.0, 15.0), bounds)

    def test_anti_meridian(self):
        ds1, ds2 = _get_antimeridian_datasets()
        bounds = get_dataset_bounds(ds1)
        self.assertEqual((165.0, -15.0, -155.0, 15.0), bounds)
        bounds = get_dataset_bounds(ds2)
        self.assertEqual((165.0, -15.0, -155.0, 15.0), bounds)


def _get_nominal_datasets():
    data_vars = dict(a=(("time", "lat", "lon"), np.random.rand(5, 3, 4)))

    coords = dict(time=(("time",), np.array(range(0, 5))),
                  lat=(("lat",), np.array([-10, 0., 10])),
                  lon=(("lon",), np.array([-20, -10, 0., 10])))
    ds1 = xr.Dataset(coords=coords, data_vars=data_vars)

    # noinspection PyTypeChecker
    coords.update(
        lat_bnds=(
            ("lat", "bnds"),
            np.array([[-15, -5], [-5., 5], [5, 15]])
        ),
        lon_bnds=(
            ("lon", "bnds"),
            np.array([[-25., -15.], [-15., -5.], [-5., 5.], [5., 15.]])
        )
    )
    ds2 = xr.Dataset(coords=coords, data_vars=data_vars)

    return ds1, ds2


def _get_inv_y_datasets():
    ds1, ds2 = _get_nominal_datasets()
    ds1 = ds1.assign_coords(lat=(("lat",), ds1.lat.values[::-1]))
    ds2 = ds2.assign_coords(lat=(("lat",), ds1.lat.values))
    ds2 = ds2.assign_coords(lat_bnds=(("lat", "bnds"),
                                      ds2.lat_bnds.values[::-1, ::-1]))
    return ds1, ds2


def _get_inv_y_wrong_order_bounds_datasets():
    ds1, ds2 = _get_nominal_datasets()
    ds1 = ds1.assign_coords(lat=(("lat",), ds1.lat.values[::-1]))
    ds2 = ds2.assign_coords(lat=(("lat",), ds1.lat.values))
    ds2 = ds2.assign_coords(lat_bnds=(("lat", "bnds"),
                                      ds2.lat_bnds.values[::-1, ::]))
    return ds1, ds2


def _get_antimeridian_datasets():
    ds1, ds2 = _get_nominal_datasets()
    ds1 = ds1.assign_coords(lon=(("lon",),
                                 np.array([170., 180., -170., -160.])))
    ds2 = ds2.assign_coords(lon=(("lon",),
                                 ds1.lon.values))
    ds2 = ds2.assign_coords(
        lon_bnds=(("lon", "bnds"),
                  np.array([[165., 175], [175., -175.],
                            [-175., -165], [-165., -155.]])))
    return ds1, ds2


class NormalizeGeometryTest(unittest.TestCase):
    def test_normalize_null(self):
        self.assertIs(None, normalize_geometry(None))

    def test_normalize_to_point(self):
        expected_point = shapely.geometry.Point(12.8, -34.4)
        self.assertIs(expected_point,
                      normalize_geometry(expected_point))
        self.assertEqual(expected_point,
                         normalize_geometry([12.8, -34.4]))
        self.assertEqual(expected_point,
                         normalize_geometry(np.array([12.8, -34.4])))
        self.assertEqual(expected_point,
                         normalize_geometry(expected_point.wkt))
        self.assertEqual(expected_point,
                         normalize_geometry(expected_point.__geo_interface__))

    def test_normalize_box_as_point(self):
        expected_point = shapely.geometry.Point(12.8, -34.4)
        self.assertEqual(expected_point,
                         normalize_geometry([12.8, -34.4, 12.8, -34.4]))

    def test_normalize_to_box(self):
        expected_box = shapely.geometry.box(12.8, -34.4, 14.2, 20.6)
        self.assertIs(expected_box,
                      normalize_geometry(expected_box))
        self.assertEqual(expected_box,
                         normalize_geometry([12.8, -34.4, 14.2, 20.6]))
        self.assertEqual(expected_box,
                         normalize_geometry(
                             np.array([12.8, -34.4, 14.2, 20.6])
                         ))
        self.assertEqual(expected_box,
                         normalize_geometry(expected_box.wkt))
        self.assertEqual(expected_box,
                         normalize_geometry(expected_box.__geo_interface__))

    def test_normalize_to_split_box(self):
        expected_split_box = shapely.geometry.MultiPolygon(polygons=[
            shapely.geometry.Polygon(
                ((180.0, -34.4), (180.0, 20.6), (172.1, 20.6), (172.1, -34.4),
                 (180.0, -34.4))
            ),
            shapely.geometry.Polygon(
                ((-165.7, -34.4), (-165.7, 20.6), (-180.0, 20.6),
                 (-180.0, -34.4),
                 (-165.7, -34.4)))]
        )
        self.assertEqual(expected_split_box,
                         normalize_geometry([172.1, -34.4, -165.7, 20.6]))

    def test_normalize_from_geo_json_feature_dict(self):
        expected_box1 = shapely.geometry.box(-10, -20, 20, 10)
        expected_box2 = shapely.geometry.box(30, 20, 50, 40)
        feature1 = dict(type='Feature',
                        geometry=expected_box1.__geo_interface__)
        feature2 = dict(type='Feature',
                        geometry=expected_box2.__geo_interface__)
        feature_collection = dict(type='FeatureCollection',
                                  features=(feature1, feature2))

        actual_geom = normalize_geometry(feature1)
        self.assertEqual(expected_box1, actual_geom)

        actual_geom = normalize_geometry(feature2)
        self.assertEqual(expected_box2, actual_geom)

        expected_geom = shapely.geometry.GeometryCollection(
            geoms=[expected_box1, expected_box2]
        )
        actual_geom = normalize_geometry(feature_collection)
        self.assertEqual(expected_geom, actual_geom)

    def test_normalize_invalid_box(self):
        from xcube.core.geom import _INVALID_BOX_COORDS_MSG

        with self.assertRaises(ValueError) as cm:
            normalize_geometry([12.8, 20.6, 14.2, -34.4])
        self.assertEqual(_INVALID_BOX_COORDS_MSG, f'{cm.exception}')
        with self.assertRaises(ValueError) as cm:
            normalize_geometry([12.8, -34.4, 12.8, 20.6])
        self.assertEqual(_INVALID_BOX_COORDS_MSG, f'{cm.exception}')
        with self.assertRaises(ValueError) as cm:
            normalize_geometry([12.8, -34.4, 12.8, 20.6])
        self.assertEqual(_INVALID_BOX_COORDS_MSG, f'{cm.exception}')

    def test_invalid(self):
        from xcube.core.geom import _INVALID_GEOMETRY_MSG

        with self.assertRaises(ValueError) as cm:
            normalize_geometry(dict(coordinates=[12.8, -34.4]))
        self.assertEqual(_INVALID_GEOMETRY_MSG, f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            normalize_geometry([12.8, -34.4, '?'])
        self.assertEqual(_INVALID_GEOMETRY_MSG, f'{cm.exception}')

    def test_deprecated(self):
        self.assertEqual(None, convert_geometry(None))


class HelpersTest(unittest.TestCase):
    def test_is_lon_lat_dataset(self):
        dataset = new_cube(x_name='lon', y_name='lat')
        self.assertTrue(is_lon_lat_dataset(dataset))

        dataset = new_cube(x_name='x', y_name='y')
        self.assertTrue(is_lon_lat_dataset(dataset))

        dataset = new_cube(x_name='x', y_name='y', x_units='meters')
        self.assertFalse(is_lon_lat_dataset(dataset))
        dataset.x.attrs.update(long_name='longitude')
        dataset.y.attrs.update(long_name='latitude')
        self.assertTrue(is_lon_lat_dataset(dataset))

    def test_is_dataset_y_axis_inverted(self):
        dataset = new_cube()
        self.assertTrue(is_dataset_y_axis_inverted(dataset))

        dataset = new_cube(inverse_y=True)
        self.assertFalse(is_dataset_y_axis_inverted(dataset))
