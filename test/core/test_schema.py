import unittest

import numpy as np
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.schema import CubeSchema


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
        self.assertEqual(('lon', 'lat'), schema.spatial_dims)

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
            CubeSchema(schema.shape[1:], schema.coords)
        self.assertEqual('shape must have at least three dimensions',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape, schema.coords, dims=('lat', 'lon'))
        self.assertEqual('dims must have same length as shape',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape, schema.coords, dims=('lat', 'lon', 'time'))
        self.assertEqual("the first name in dims must be 'time'",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape, schema.coords, dims=('time', 'lon', 'lat'))
        self.assertEqual("the last two names in dims must be either ('lat', 'lon') or ('y', 'x')",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            CubeSchema(schema.shape, schema.coords, dims=schema.dims, chunks=(90, 90))
        self.assertEqual("chunks must have same length as shape",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            coords = dict(schema.coords)
            del coords['lat']
            CubeSchema(schema.shape, coords, dims=schema.dims, chunks=(1, 90, 90))
        self.assertEqual("missing dimension 'lat' in coords",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            coords = dict(schema.coords)
            lat = coords['lat']
            coords['lat'] = xr.DataArray(lat.values.reshape((1, len(lat))), dims=('b', lat.dims[0]), attrs=lat.attrs)
            CubeSchema(schema.shape, coords, dims=schema.dims, chunks=(1, 90, 90))
        self.assertEqual("labels of 'lat' in coords must be one-dimensional",
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

