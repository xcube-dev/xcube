import unittest

from xcube.core.gen2.rechunker import CubeRechunker
from xcube.core.new import new_cube


class CubeRechunkerTest(unittest.TestCase):

    def test_chunks_are_smaller_than_sizes(self):
        cube1 = new_cube(variables=dict(chl=0.6, tsm=0.9, flags=16))
        for var in cube1.variables.values():
            self.assertIsNone(var.chunks)

        rc = CubeRechunker(chunks=dict(time=2, lat=100, lon=200))
        cube2 = rc.process_cube(cube1)

        self.assertEqual(cube1.attrs, cube2.attrs)
        self.assertEqual(set(cube1.coords), set(cube2.coords))
        self.assertEqual(set(cube1.data_vars), set(cube2.data_vars))

        for k, v in cube2.coords.items():
            if v.chunks is not None:
                self.assertIsInstance(v.chunks, tuple, msg=f'{k!r}={v!r}')
                self.assertIn('chunks', v.encoding)
                self.assertEqual([v.sizes[d] for d in v.dims],
                                 v.encoding['chunks'])
        for k, v in cube2.data_vars.items():
            self.assertIsInstance(v.chunks, tuple, msg=f'{k!r}={v!r}')
            self.assertEqual(((2, 2, 1), (100, 80), (200, 160)), v.chunks)
            self.assertIn('chunks', v.encoding)
            self.assertEqual([2, 100, 200], v.encoding['chunks'])

    def test_chunks_are_larger_than_sizes(self):
        cube1 = new_cube(variables=dict(chl=0.6, tsm=0.9, flags=16))
        for var in cube1.variables.values():
            self.assertIsNone(var.chunks)

        rc = CubeRechunker(chunks=dict(time=64, lat=512, lon=512))
        cube2 = rc.process_cube(cube1)

        self.assertEqual(cube1.attrs, cube2.attrs)
        self.assertEqual(set(cube1.coords), set(cube2.coords))
        self.assertEqual(set(cube1.data_vars), set(cube2.data_vars))

        for k, v in cube2.coords.items():
            if v.chunks is not None:
                self.assertIsInstance(v.chunks, tuple, msg=f'{k!r}={v!r}')
                self.assertIn('chunks', v.encoding)
                self.assertEqual([v.sizes[d] for d in v.dims],
                                 v.encoding['chunks'])
        for k, v in cube2.data_vars.items():
            self.assertIsInstance(v.chunks, tuple, msg=f'{k!r}={v!r}')
            self.assertEqual(((5,), (180,), (360,)), v.chunks)
            self.assertIn('chunks', v.encoding)
            self.assertEqual([5, 180, 360], v.encoding['chunks'])
