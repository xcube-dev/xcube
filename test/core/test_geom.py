import unittest

import numpy as np
import shapely.geometry
import xarray as xr

from xcube.core.chunk import chunk_dataset
from xcube.core.geom import get_dataset_geometry, get_dataset_bounds, get_geometry_mask, convert_geometry, \
    mask_dataset_by_geometry, clip_dataset_by_geometry, rasterize_features_into_dataset
from xcube.core.new import new_cube


class RasterizeFeaturesIntoDataset(unittest.TestCase):
    def test_rasterize_features_into_dataset(self):
        dataset = new_cube(width=10, height=10, spatial_res=10, lon_start=-50, lat_start=-50)
        feature1 = dict(type='Feature',
                        geometry=dict(type='Polygon',
                                      coordinates=[[(-180, 0), (-1, 0), (-1, 90), (-180, 90), (-180, 0)]]),
                        properties=dict(a=0.5, b=2.1, c=4.9))
        feature2 = dict(type='Feature',
                        geometry=dict(type='Polygon',
                                      coordinates=[[(-180, -90), (-1, -90), (-1, 0), (-180, 0), (-180, -90)]]),
                        properties=dict(a=0.6, b=2.2, c=4.8))
        feature3 = dict(type='Feature',
                        geometry=dict(type='Polygon',
                                      coordinates=[[(20, 0), (180, 0), (180, 90), (20, 90), (20, 0)]]),
                        properties=dict(a=0.7, b=2.3, c=4.7))
        feature4 = dict(type='Feature',
                        geometry=dict(type='Polygon',
                                      coordinates=[[(20, -90), (180, -90), (180, 0), (20, 0), (20, -90)]]),
                        properties=dict(a=0.8, b=2.4, c=4.6))
        dataset = rasterize_features_into_dataset(dataset,
                                                  [feature1, feature2, feature3, feature4],
                                                  ['a', 'b', 'c'],
                                                  in_place=False)
        self.assertIsNotNone(dataset)
        self.assertIn('lon', dataset.coords)
        self.assertIn('lat', dataset.coords)
        self.assertIn('time', dataset.coords)
        self.assertIn('a', dataset)
        self.assertIn('b', dataset)
        self.assertIn('c', dataset)
        self.assertEquals((10, 10), dataset.a.shape)
        self.assertEquals((10, 10), dataset.b.shape)
        self.assertEquals((10, 10), dataset.c.shape)
        self.assertEquals(('lat', 'lon'), dataset.a.dims)
        self.assertEquals(('lat', 'lon'), dataset.b.dims)
        self.assertEquals(('lat', 'lon'), dataset.c.dims)
        self.assertIn(0.5, dataset.a.min())
        self.assertIn(0.8, dataset.a.max())
        self.assertIn(2.1, dataset.b.min())
        self.assertIn(2.4, dataset.b.max())
        self.assertIn(4.6, dataset.c.min())
        self.assertIn(4.9, dataset.c.max())
        nan = np.nan
        np.testing.assert_almost_equal(
            np.array([[0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
                      [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
                      [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
                      [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
                      [0.5, 0.5, 0.5, 0.5, 0.5, nan, nan, 0.7, 0.7, 0.7],
                      [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
                      [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
                      [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
                      [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8],
                      [0.6, 0.6, 0.6, 0.6, 0.6, nan, nan, 0.8, 0.8, 0.8]]),
            dataset.a.values)


class DatasetGeometryTest(unittest.TestCase):

    def setUp(self) -> None:
        width = 16
        height = 8
        spatial_res = 360 / width
        lon_min = -2 * spatial_res + 0.5 * spatial_res
        lat_min = -2 * spatial_res + 0.5 * spatial_res
        lon_max = lon_min + 6 * spatial_res
        lat_max = lat_min + 3 * spatial_res

        self.triangle = shapely.geometry.Polygon(((lon_min, lat_min),
                                                  (lon_max, lat_min),
                                                  (0.5 * (lon_max + lon_min), lat_max),
                                                  (lon_min, lat_min)))

        self.cube = new_cube(width=width,
                             height=height,
                             spatial_res=spatial_res,
                             drop_bounds=True,
                             variables=dict(temp=273.9, precip=0.9))

    def test_clip_dataset_by_geometry(self):
        cube = clip_dataset_by_geometry(self.cube, self.triangle)
        self._assert_clipped_dataset_has_basic_props(cube)
        cube = clip_dataset_by_geometry(self.cube, self.triangle, save_geometry_wkt=True)
        self._assert_clipped_dataset_has_basic_props(cube)
        self._assert_saved_geometry_wkt_is_fine(cube, 'geometry_wkt')
        cube = clip_dataset_by_geometry(self.cube, self.triangle, save_geometry_wkt='intersect_geom')
        self._assert_saved_geometry_wkt_is_fine(cube, 'intersect_geom')

    def test_mask_dataset_by_geometry(self):
        cube = mask_dataset_by_geometry(self.cube, self.triangle)
        self._assert_clipped_dataset_has_basic_props(cube)
        cube = mask_dataset_by_geometry(self.cube, self.triangle, save_geometry_wkt=True)
        self._assert_saved_geometry_wkt_is_fine(cube, 'geometry_wkt')
        cube = mask_dataset_by_geometry(self.cube, self.triangle, save_geometry_wkt='intersect_geom')
        self._assert_saved_geometry_wkt_is_fine(cube, 'intersect_geom')

    def test_mask_dataset_by_geometry_excluded_vars(self):
        cube = mask_dataset_by_geometry(self.cube, self.triangle, excluded_vars='precip')
        self._assert_clipped_dataset_has_basic_props(cube)

    def test_mask_dataset_by_geometry_store_mask(self):
        cube = mask_dataset_by_geometry(self.cube, self.triangle, save_geometry_mask='geom_mask')
        self._assert_clipped_dataset_has_basic_props(cube)
        self._assert_dataset_mask_is_fine(cube, 'geom_mask')

    def test_clip_dataset_for_chunked_input(self):
        cube = chunk_dataset(self.cube, chunk_sizes=dict(time=1, lat=90, lon=90))
        cube = clip_dataset_by_geometry(cube, self.triangle)
        self._assert_clipped_dataset_has_basic_props(cube)
        self.assertEqual(((1, 1, 1, 1, 1), (4,), (7,)), cube.temp.chunks)
        self.assertEqual(((1, 1, 1, 1, 1), (4,), (7,)), cube.precip.chunks)

    def test_mask_dataset_for_chunked_input(self):
        cube = chunk_dataset(self.cube, chunk_sizes=dict(time=1, lat=90, lon=90))
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
        np.testing.assert_array_almost_equal(geom_mask.values.astype(np.byte),
                                             np.array([[0, 0, 0, 1, 0, 0, 0],
                                                       [0, 0, 1, 1, 1, 0, 0],
                                                       [0, 1, 1, 1, 1, 1, 0],
                                                       [1, 1, 1, 1, 1, 1, 1]], dtype=np.byte))

    def _assert_saved_geometry_wkt_is_fine(self, dataset, geometry_wkt_name):
        self.assertIn(geometry_wkt_name, dataset.attrs)
        self.assertEqual('POLYGON ((-33.75 -33.75, 33.75 33.75, 101.25 -33.75, -33.75 -33.75))',
                         dataset.attrs[geometry_wkt_name])


class GetGeometryMaskTest(unittest.TestCase):
    def test_get_geometry_mask(self):
        w = 16
        h = 8
        res = 1.0
        lon_min = 0
        lat_min = 0
        lon_max = lon_min + w * res
        lat_max = lat_min + h * res

        triangle = shapely.geometry.Polygon(((lon_min, lat_min), (lon_max, lat_min), (lon_max, lat_max),
                                             (lon_min, lat_min)))

        actual_mask = get_geometry_mask(w, h, triangle, lon_min, lat_min, res)
        expected_mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.byte)
        np.testing.assert_array_almost_equal(expected_mask, actual_mask.astype('byte'))

        smaller_triangle = triangle.buffer(-1.5 * res)
        actual_mask = get_geometry_mask(w, h, smaller_triangle, lon_min, lat_min, res)
        expected_mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.byte)
        np.testing.assert_array_almost_equal(expected_mask, actual_mask)


class GetDatasetGeometryTest(unittest.TestCase):

    def test_nominal(self):
        ds1, ds2 = _get_nominal_datasets()
        bounds = get_dataset_geometry(ds1)
        self.assertEqual(shapely.geometry.box(-25.0, -15.0, 15.0, 15.0), bounds)
        bounds = get_dataset_geometry(ds2)
        self.assertEqual(shapely.geometry.box(-25.0, -15.0, 15.0, 15.0), bounds)

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

    coords.update(dict(lat_bnds=(("lat", "bnds"), np.array([[-15, -5], [-5., 5], [5, 15]])),
                       lon_bnds=(
                           ("lon", "bnds"), np.array([[-25., -15.], [-15., -5.], [-5., 5.], [5., 15.]]))
                       ))
    ds2 = xr.Dataset(coords=coords, data_vars=data_vars)

    return ds1, ds2


def _get_antimeridian_datasets():
    ds1, ds2 = _get_nominal_datasets()
    ds1 = ds1.assign_coords(lon=(("lon",), np.array([170., 180., -170., -160.])))
    ds2 = ds2.assign_coords(
        lon_bnds=(("lon", "bnds"), np.array([[165., 175], [175., -175.], [-175., -165], [-165., -155.]])))
    return ds1, ds2


class ConvertGeometryTest(unittest.TestCase):
    def test_convert_null(self):
        self.assertIs(None, convert_geometry(None))

    def test_convert_to_point(self):
        expected_point = shapely.geometry.Point(12.8, -34.4)
        self.assertIs(expected_point,
                      convert_geometry(expected_point))
        self.assertEqual(expected_point,
                         convert_geometry([12.8, -34.4]))
        self.assertEqual(expected_point,
                         convert_geometry(np.array([12.8, -34.4])))
        self.assertEqual(expected_point,
                         convert_geometry(expected_point.wkt))
        self.assertEqual(expected_point,
                         convert_geometry(expected_point.__geo_interface__))

    def test_convert_box_as_point(self):
        expected_point = shapely.geometry.Point(12.8, -34.4)
        self.assertEqual(expected_point,
                         convert_geometry([12.8, -34.4, 12.8, -34.4]))

    def test_convert_to_box(self):
        expected_box = shapely.geometry.box(12.8, -34.4, 14.2, 20.6)
        self.assertIs(expected_box,
                      convert_geometry(expected_box))
        self.assertEqual(expected_box,
                         convert_geometry([12.8, -34.4, 14.2, 20.6]))
        self.assertEqual(expected_box,
                         convert_geometry(np.array([12.8, -34.4, 14.2, 20.6])))
        self.assertEqual(expected_box,
                         convert_geometry(expected_box.wkt))
        self.assertEqual(expected_box,
                         convert_geometry(expected_box.__geo_interface__))

    def test_convert_to_split_box(self):
        expected_split_box = shapely.geometry.MultiPolygon(polygons=[
            shapely.geometry.Polygon(((180.0, -34.4), (180.0, 20.6), (172.1, 20.6), (172.1, -34.4),
                                      (180.0, -34.4))),
            shapely.geometry.Polygon(((-165.7, -34.4), (-165.7, 20.6), (-180.0, 20.6), (-180.0, -34.4),
                                      (-165.7, -34.4)))])
        self.assertEqual(expected_split_box,
                         convert_geometry([172.1, -34.4, -165.7, 20.6]))

    def test_convert_from_geojson_feature_dict(self):
        expected_box1 = shapely.geometry.box(-10, -20, 20, 10)
        expected_box2 = shapely.geometry.box(30, 20, 50, 40)
        feature1 = dict(type='Feature', geometry=expected_box1.__geo_interface__)
        feature2 = dict(type='Feature', geometry=expected_box2.__geo_interface__)
        feature_collection = dict(type='FeatureCollection', features=(feature1, feature2))

        actual_geom = convert_geometry(feature1)
        self.assertEqual(expected_box1, actual_geom)

        actual_geom = convert_geometry(feature2)
        self.assertEqual(expected_box2, actual_geom)

        expected_geom = shapely.geometry.GeometryCollection(geoms=[expected_box1, expected_box2])
        actual_geom = convert_geometry(feature_collection)
        self.assertEqual(expected_geom, actual_geom)

    def test_convert_invalid_box(self):
        from xcube.core.geom import _INVALID_BOX_COORDS_MSG

        with self.assertRaises(ValueError) as cm:
            convert_geometry([12.8, 20.6, 14.2, -34.4])
        self.assertEqual(_INVALID_BOX_COORDS_MSG, f'{cm.exception}')
        with self.assertRaises(ValueError) as cm:
            convert_geometry([12.8, -34.4, 12.8, 20.6])
        self.assertEqual(_INVALID_BOX_COORDS_MSG, f'{cm.exception}')
        with self.assertRaises(ValueError) as cm:
            convert_geometry([12.8, -34.4, 12.8, 20.6])
        self.assertEqual(_INVALID_BOX_COORDS_MSG, f'{cm.exception}')

    def test_invalid(self):
        from xcube.core.geom import _INVALID_GEOMETRY_MSG

        with self.assertRaises(ValueError) as cm:
            convert_geometry(dict(coordinates=[12.8, -34.4]))
        self.assertEqual(_INVALID_GEOMETRY_MSG, f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            convert_geometry([12.8, -34.4, '?'])
        self.assertEqual(_INVALID_GEOMETRY_MSG, f'{cm.exception}')
