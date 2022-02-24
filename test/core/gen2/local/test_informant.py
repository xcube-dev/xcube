import unittest
from typing import Dict, Any

from xcube.core.gen2.config import CubeConfig
from xcube.core.gen2.config import InputConfig
from xcube.core.gen2.config import OutputConfig
from xcube.core.gen2.local.informant import CubeInformant
from xcube.core.gen2.request import CubeGeneratorRequest
from xcube.core.store import DataStorePool
from xcube.core.store import DataStoreConfig
from xcube.core.gen2.response import CubeInfo
from xcube.core.new import new_cube

class InformantTest(unittest.TestCase):

    def setUp(self) -> None:
        self.store_pool = DataStorePool(dict(
            mem=DataStoreConfig(
                store_id="memory",
            )
        ))
        cube = new_cube(variables=dict(B01=0.1, B02=0.2, B03=0.3))
        self.store_pool.get_store('mem').write_data(
            cube,
            'informant_test.zarr',
            replace=True,
        )
        self.input_config = InputConfig(
            store_id='memory',
            data_id='informant_test.zarr'
        )
        self.output_config = OutputConfig(
            store_id='memory',
            data_id='output.zarr',
            replace=True
        )

    def test_effective_cube_config_1(self):
        cube_config = CubeConfig(
            variable_names=['B01', 'B03'],
            crs=None,
            bbox=(12.2, 52.1, 13.9, 54.8),
            spatial_res=(0.05, 0.1),
            time_range=('2010-01-01', None),
            time_period='4D',
            chunks=dict(time=None, lat=90, lon=90),
            metadata=dict(title='A S2L2A subset')
        )
        request = CubeGeneratorRequest(
            input_config=self.input_config,
            cube_config=cube_config,
            output_config=self.output_config
        )

        informant = CubeInformant(request=request, store_pool=self.store_pool)

        effective_cube_config = informant.effective_cube_config

        self.assertIsInstance(effective_cube_config, CubeConfig)
        self.assertEqual((12.2, 52.1, 13.9, 54.8), effective_cube_config.bbox)
        self.assertEqual('WGS84', effective_cube_config.crs)
        self.assertEqual((0.05, 0.1), effective_cube_config.spatial_res)
        self.assertEqual('4D', effective_cube_config.time_period)
        self.assertEqual(('2010-01-01', '2010-01-06'),
                         effective_cube_config.time_range)
        self.assertEqual(('B01', 'B03'),
                         effective_cube_config.variable_names)

    def test_effective_cube_config_2(self):
        cube_config = CubeConfig(
            crs='EPSG:6931',
            time_range=('2010-01-01', '2010-12-31'),
            metadata=dict(title='A S2L2A subset')
        )
        request = CubeGeneratorRequest(
            input_config=self.input_config,
            cube_config=cube_config,
            output_config=self.output_config
        )

        informant = CubeInformant(request=request, store_pool=self.store_pool)

        effective_cube_config = informant.effective_cube_config

        self.assertIsInstance(effective_cube_config, CubeConfig)
        self.assertEqual((-180.0, -90.0, 180.0, 90.0),
                         effective_cube_config.bbox)
        self.assertEqual('EPSG:6931', effective_cube_config.crs)
        self.assertEqual(1.0, effective_cube_config.spatial_res)
        self.assertEqual('1D', effective_cube_config.time_period)
        self.assertEqual(('2010-01-01', '2010-12-31'),
                         effective_cube_config.time_range)
        self.assertEqual(('B01', 'B02', 'B03'),
                         effective_cube_config.variable_names)

    def test_generate(self):
        cube_config = CubeConfig(
            variable_names=['B01', 'B03'],
            crs=None,
            bbox=(12.2, 52.1, 13.9, 54.8),
            spatial_res=(0.05, 0.1),
            time_range=('2010-01-01', None),
            time_period='4D',
            chunks=dict(time=None, lat=90, lon=90),
            metadata=dict(title='A S2L2A subset')
        )
        request = CubeGeneratorRequest(
            input_config=self.input_config,
            cube_config=cube_config,
            output_config=self.output_config
        )

        informant = CubeInformant(request=request, store_pool=self.store_pool)
        cube_info = informant.generate()
        self.assertIsInstance(cube_info, CubeInfo)
        self.assertIsNotNone(cube_info.dataset_descriptor)
        self.assertIsNotNone(cube_info.size_estimation)
        self.assertEqual([34, 27], cube_info.size_estimation['image_size'])
        self.assertEqual([34, 27], cube_info.size_estimation['tile_size'])
        self.assertEqual(2, cube_info.size_estimation['num_variables'])
        self.assertEqual([1, 1], cube_info.size_estimation['num_tiles'])
        self.assertEqual(4, cube_info.size_estimation['num_requests'])
        self.assertEqual(14688, cube_info.size_estimation['num_bytes'])
