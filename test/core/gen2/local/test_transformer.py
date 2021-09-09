import unittest

import xarray as xr

from xcube.core.gen2 import CubeConfig
from xcube.core.gen2.local.transformer import CubeIdentity
from xcube.core.gen2.local.transformer import CubeTransformer
from xcube.core.gen2.local.transformer import TransformedCube
from xcube.core.gen2.local.transformer import transform_cube
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube

CALLBACK_MOCK_URL = 'https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback'


class CubeIdentityTest(unittest.TestCase):

    def test_it(self):
        cube = new_cube(variables=dict(a=0.5))
        gm = GridMapping.from_dataset(cube)
        cube_config = CubeConfig()
        identity = CubeIdentity()
        t_cube = identity.transform_cube(cube,
                                         gm,
                                         cube_config)
        self.assertIsInstance(t_cube, tuple)
        self.assertEqual(3, len(t_cube))
        self.assertIs(cube, t_cube[0])
        self.assertIs(gm, t_cube[1])
        self.assertIs(cube_config, t_cube[2])


class MyTiler(CubeTransformer):

    def transform_cube(self,
                       cube: xr.Dataset,
                       gm: GridMapping,
                       cube_config: CubeConfig) -> TransformedCube:
        cube = cube.chunk(dict(lon=cube_config.tile_size[0],
                               lat=cube_config.tile_size[1]))
        cube_config = cube_config.drop_props('tile_size')
        return cube, gm, cube_config


class TransformCubeTest(unittest.TestCase):

    def test_non_empty_cube(self):
        cube = new_cube(variables=dict(a=0.5))
        gm = GridMapping.from_dataset(cube)
        cube_config = CubeConfig(tile_size=180)

        t_cube = transform_cube((cube, gm, cube_config), MyTiler())
        self.assertIsInstance(t_cube, tuple)
        self.assertEqual(3, len(t_cube))

        cube2, gm2, cc2 = t_cube
        self.assertIsNot(cube, cube2)
        self.assertEqual(((5,), (180,), (180, 180)), cube2.a.chunks)
        self.assertIs(gm, gm2)
        self.assertEqual(None, cc2.tile_size)

    def test_empty_cube(self):
        cube = new_cube()
        gm = GridMapping.from_dataset(cube)
        cube_config = CubeConfig(tile_size=180)

        t_cube = transform_cube((cube, gm, cube_config), MyTiler())
        self.assertIsInstance(t_cube, tuple)
        self.assertEqual(3, len(t_cube))

        cube2, gm2, cc2 = t_cube
        self.assertIs(cube, cube2)
        self.assertIs(gm, gm2)
        self.assertIs(cube_config, cc2)
        self.assertEqual((180, 180), cc2.tile_size)
