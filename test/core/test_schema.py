import unittest

import numpy as np
import pytest
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.schema import CubeSchema
from xcube.core.schema import get_dataset_chunks
from xcube.core.schema import get_dataset_xy_var_names


class CubeSchemaTest(unittest.TestCase):

    def test_new(self):
        cube = new_cube(variables=dict(a=2, b=3, c=4))
        schema = CubeSchema.new(cube)
        self._assert_schema(schema, expected_shape=cube.a.shape)

    def test_new_chunked(self):
        cube = new_cube(variables=dict(a=2, b=3, c=4))
        cube = cube.chunk(dict(time=1, lat=90, lon=90))
        schema = CubeSchema.new(cube)
        self._assert_schema(schema, expected_shape=cube.a.shape, expected_chunks=(1, 90, 90))

    def test_xarray_accessor(self):
        # noinspection PyUnresolvedReferences
        import xcube.core.xarray
        cube = new_cube(variables=dict(a=2, b=3, c=4))
        schema = cube.xcube.schema
        self._assert_schema(schema, expected_shape=cube.a.shape)

    def _assert_schema(self, schema: CubeSchema, expected_shape=None, expected_chunks=None):
        self.assertEqual(3, schema.ndim)
        self.assertEqual(expected_shape, schema.shape)
        self.assertEqual(expected_chunks, schema.chunks)
        self.assertIn('time', schema.coords)
        self.assertIn('lat', schema.coords)
        self.assertIn('lon', schema.coords)
        self.assertEqual(('time', 'lat', 'lon'), schema.dims)
        self.assertEqual('time', schema.time_dim)
        self.assertEqual('lon', schema.x_name)
        self.assertEqual('lat', schema.y_name)
        self.assertEqual('time', schema.time_name)
        self.assertEqual('lon', schema.x_dim)
        self.assertEqual('lat', schema.y_dim)
        self.assertEqual('time', schema.time_dim)
        self.assertEqual(expected_shape[-1], schema.x_size)
        self.assertEqual(expected_shape[-2], schema.y_size)
        self.assertEqual(expected_shape[0], schema.time_size)
        self.assertEqual(expected_shape[-1], schema.x_var.size)
        self.assertEqual(expected_shape[-2], schema.y_var.size)
        self.assertEqual(expected_shape[0], schema.time_var.size)

    def test_repr_html(self):
        cube = new_cube(variables=dict(a=2, b=3, c=4))
        cube = cube.chunk(dict(time=1, lat=90, lon=90))
        schema = CubeSchema.new(cube)
        self.assertEqual("<table>"
                         "<tr><td>Shape:</td><td>(5, 180, 360)</td></tr>"
                         "<tr><td>Chunk sizes:</td><td>(1, 90, 90)</td></tr>"
                         "<tr><td>Dimensions:</td><td>('time', 'lat', 'lon')</td></tr>"
                         "</table>",
                         schema._repr_html_())

    def test_constructor_with_invalid_args(self):
        cube = new_cube(variables=dict(t=273))
        schema = CubeSchema.new(cube)
        with self.assertRaises(ValueError) as cm:
            # noinspection PyTypeChecker
            CubeSchema(None, schema.coords)
        self.assertEqual('shape must be a sequence of integer sizes',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            # noinspection PyTypeChecker
            CubeSchema(schema.shape, None)
        self.assertEqual('coords must be a mapping from dimension names to label arrays',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            # noinspection PyTypeChecker
            CubeSchema(schema.shape, cube.coords, x_name=None)
        self.assertEqual('x_name must be given',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            # noinspection PyTypeChecker
            CubeSchema(schema.shape, cube.coords, y_name=None)
        self.assertEqual('y_name must be given',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            # noinspection PyTypeChecker
            CubeSchema(schema.shape, cube.coords, time_name=None)
        self.assertEqual('time_name must be given',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape[1:], schema.coords)
        self.assertEqual('shape must have at least three dimensions',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape, schema.coords, dims=('lat', 'lon'))
        self.assertEqual('dims must have same length as shape',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape, schema.coords, dims=('lat', 'lon', 'time'))
        self.assertEqual("the first dimension in dims must be 'time'",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape, schema.coords, dims=('time', 'lon', 'lat'))
        self.assertEqual("the last two dimensions in dims must be 'lat' and 'lon'",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape, schema.coords, dims=schema.dims, chunks=(90, 90))
        self.assertEqual("chunks must have same length as shape",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            coords = dict(schema.coords)
            del coords['lat']
            CubeSchema(schema.shape, coords, dims=schema.dims, chunks=(1, 90, 90))
        self.assertEqual("missing variables 'lon', 'lat', 'time' in coords",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            coords = dict(schema.coords)
            lat = coords['lat']
            coords['lat'] = xr.DataArray(lat.values.reshape((1, len(lat))), dims=('b', lat.dims[0]), attrs=lat.attrs)
            CubeSchema(schema.shape, coords, dims=schema.dims, chunks=(1, 90, 90))
        self.assertEqual("variables 'lon', 'lat', 'time' in coords must be 1-D",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            coords = dict(schema.coords)
            lat = coords['lat']
            coords['lat'] = xr.DataArray(lat.values[1:], dims=('lat',), attrs=lat.attrs)
            CubeSchema(schema.shape, coords, dims=schema.dims, chunks=(1, 90, 90))
        self.assertEqual("number of labels of 'lat' in coords does not match shape",
                         f'{cm.exception}')

    def test_new_with_cube(self):
        cube = new_cube()
        with self.assertRaises(ValueError) as cm:
            CubeSchema.new(cube)
        self.assertEqual("cube is empty",
                         f'{cm.exception}')

        cube = new_cube()
        del cube.coords['lon']
        with self.assertRaises(ValueError) as cm:
            CubeSchema.new(cube)
        self.assertEqual("cube has no valid spatial coordinate variables",
                         f'{cm.exception}')

        cube = new_cube()
        del cube.coords['time']
        with self.assertRaises(ValueError) as cm:
            CubeSchema.new(cube)
        self.assertEqual("cube has no valid time coordinate variable",
                         f'{cm.exception}')

        cube = new_cube(variables=dict(a=1, b=2))
        cube['c'] = xr.DataArray(np.array([1, 2, 3, 4, 5]), dims=('q',))
        with self.assertRaises(ValueError) as cm:
            CubeSchema.new(cube)
        self.assertEqual("all variables must have same dimensions, but variable 'c' has dimensions ('q',)",
                         f'{cm.exception}')

        cube = new_cube(variables=dict(a=1, b=2))
        cube = cube.chunk(dict(time=1, lat=90, lon=90))
        cube['b'] = cube['b'].chunk(dict(time=1, lat=45, lon=90))
        with self.assertRaises(ValueError) as cm:
            CubeSchema.new(cube)
        self.assertEqual("all variables must have same chunks, but variable 'b' has chunks (1, 45, 90)",
                         f'{cm.exception}')

        cube = new_cube(variables=dict(a=1, b=2))
        cube = cube.chunk(dict(time=1, lat=(44, 43, 46, 47), lon=90))
        with self.assertRaises(ValueError) as cm:
            CubeSchema.new(cube)
        self.assertEqual("dimension 'lat' of variable 'a' has chunks of different sizes: (44, 43, 46, 47)",
                         f'{cm.exception}')


class GetDatasetChunksTest(unittest.TestCase):

    def test_empty_dataset(self):
        dataset = xr.Dataset()
        self.assertEqual({}, get_dataset_chunks(dataset))

    def test_only_coords_dataset(self):
        dataset = xr.Dataset(
            coords=dict(
                x=xr.DataArray(np.linspace(0, 1, 100), dims='x').chunk(10),
                y=xr.DataArray(np.linspace(0, 1, 100), dims='y').chunk(20),
            )
        )
        self.assertEqual({}, get_dataset_chunks(dataset))

    def test_different_chunks(self):
        dataset = xr.Dataset(data_vars=dict(
            a=xr.DataArray(
                np.linspace(0, 1, 100 * 100).reshape((100, 100)),
                dims=('y', 'x')
            ).chunk((30, 10)),
            b=xr.DataArray(
                np.linspace(0, 1, 100 * 100).reshape((100, 100)),
                dims=('y', 'x')
            ).chunk((25, 15)),
            c=xr.DataArray(
                np.linspace(0, 1, 100 * 100).reshape((1, 100, 100)),
                dims=('time', 'y', 'x')
            ).chunk((1, 25, 10)),
            d=xr.DataArray(  # d is not chunked!
                np.linspace(0, 1, 100 * 100).reshape((1, 100, 100)),
                dims=('time', 'y', 'x')
            ),
        ))
        self.assertEqual(
            {'time': 1, 'x': 10, 'y': 25},
            get_dataset_chunks(dataset)
        )


class GetDatasetXYVarNamesTest(unittest.TestCase):

    def test_find_by_cf_standard_name_attr(self):
        dataset = xr.Dataset(coords=dict(
            u=xr.DataArray([1, 2, 3], attrs=dict(
                standard_name="projection_x_coordinate"
            )),
            v=xr.DataArray([1, 2, 3], attrs=dict(
                standard_name="projection_y_coordinate"
            )),
        ))
        self.assertEqual(("u", "v"),
                         get_dataset_xy_var_names(dataset))

    def test_find_by_cf_long_name_attr(self):

        dataset = xr.Dataset(coords=dict(
            u=xr.DataArray([1, 2, 3], attrs=dict(
                long_name="x coordinate of projection"
            )),
            v=xr.DataArray([1, 2, 3], attrs=dict(
                long_name="y coordinate of projection"
            )),
        ))
        self.assertEqual(("u", "v"),
                         get_dataset_xy_var_names(dataset))

        dataset = xr.Dataset(coords=dict(
            a=xr.DataArray([1, 2, 3], attrs=dict(
                long_name="longitude"
            )),
            b=xr.DataArray([1, 2, 3], attrs=dict(
                long_name="latitude"
            )),
        ))
        self.assertEqual(("a", "b"),
                         get_dataset_xy_var_names(dataset))

    def test_find_by_var_name(self):

        dataset = xr.Dataset(coords=dict(
            x=xr.DataArray([1, 2, 3]),
            y=xr.DataArray([1, 2, 3]),
        ))
        self.assertEqual(("x", "y"),
                         get_dataset_xy_var_names(dataset))

        dataset = xr.Dataset(coords=dict(
            lon=xr.DataArray([1, 2, 3]),
            lat=xr.DataArray([1, 2, 3]),
        ))
        self.assertEqual(("lon", "lat"),
                         get_dataset_xy_var_names(dataset))

        dataset = xr.Dataset(coords=dict(
            longitude=xr.DataArray([1, 2, 3]),
            latitude=xr.DataArray([1, 2, 3]),
        ))
        self.assertEqual(("longitude", "latitude"),
                         get_dataset_xy_var_names(dataset))

    def test_not_found(self):
        dataset = xr.Dataset(coords=dict(
            hanni=xr.DataArray([1, 2, 3]),
            nanni=xr.DataArray([1, 2, 3]),
        ))

        self.assertIsNone(get_dataset_xy_var_names(dataset))

        with pytest.raises(ValueError,
                           match="dataset has no valid"
                                 " spatial coordinate variables"):
            get_dataset_xy_var_names(dataset, must_exist=True)

        self.assertIsNone(get_dataset_xy_var_names(dataset))
