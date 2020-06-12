import unittest

from test.webapi.helpers import new_test_service_context, RequestParamsMock
from xcube.webapi.context import ServiceContext
from xcube.webapi.controllers.tiles import get_dataset_tile, get_ne2_tile, get_dataset_tile_grid, get_ne2_tile_grid, \
    get_legend
from xcube.webapi.errors import ServiceBadRequestError, ServiceResourceNotFoundError


class TilesControllerTest(unittest.TestCase):

    def test_get_dataset_tile(self):
        ctx = new_test_service_context()
        tile = get_dataset_tile(ctx, 'demo', 'conc_tsm', '0', '0', '0', RequestParamsMock())
        self.assertIsInstance(tile, bytes)

        tile = get_dataset_tile(ctx, 'demo', 'conc_tsm', '-20', '0', '0', RequestParamsMock())
        self.assertIsInstance(tile, bytes)

    def test_get_dataset_tile_invalid_dataset(self):
        ctx = new_test_service_context()
        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            get_dataset_tile(ctx, 'demo-rgb', 'conc_tsm', '0', '0', '0', RequestParamsMock())
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Dataset "demo-rgb" not found', f'{cm.exception.reason}')

    def test_get_dataset_tile_invalid_variable(self):
        ctx = new_test_service_context()
        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            get_dataset_tile(ctx, 'demo', 'conc_tdi', '0', '0', '0', RequestParamsMock())
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Variable "conc_tdi" not found in dataset "demo"', f'{cm.exception.reason}')

    def test_get_dataset_tile_with_all_params(self):
        ctx = new_test_service_context()
        tile = get_dataset_tile(ctx, 'demo', 'conc_tsm', '0', '0', '0', RequestParamsMock(time='current', cbar='plasma',
                                                                                          vmin='0.1', vmax='0.3'))
        self.assertIsInstance(tile, bytes)

    def test_get_dataset_tile_with_time_dim(self):
        ctx = new_test_service_context()
        tile = get_dataset_tile(ctx, 'demo', 'conc_tsm', '0', '0', '0', RequestParamsMock(time='2017-01-26'))
        self.assertIsInstance(tile, bytes)

        ctx = new_test_service_context()
        tile = get_dataset_tile(ctx, 'demo', 'conc_tsm', '0', '0', '0', RequestParamsMock(time='2017-01-26/2017-01-27'))
        self.assertIsInstance(tile, bytes)

        ctx = new_test_service_context()
        tile = get_dataset_tile(ctx, 'demo', 'conc_tsm', '0', '0', '0', RequestParamsMock(time='current'))
        self.assertIsInstance(tile, bytes)

    def test_get_dataset_tile_with_invalid_time_dim(self):
        ctx = new_test_service_context()
        with self.assertRaises(ServiceBadRequestError) as cm:
            get_dataset_tile(ctx, 'demo', 'conc_tsm', '0', '0', '0', RequestParamsMock(time='Gnaaark!'))
        self.assertEqual(400, cm.exception.status_code)
        self.assertEqual("'Gnaaark!' is not a valid value for dimension 'time'",
                         cm.exception.reason)

    def test_get_dataset_rgb_tile(self):
        ctx = new_test_service_context('config-rgb.yml')
        tile = get_dataset_tile(ctx, 'demo-rgb', 'rgb', '0', '0', '0', RequestParamsMock())
        self.assertIsInstance(tile, bytes)

    def test_get_dataset_rgb_tile_invalid_b(self):
        ctx = new_test_service_context('config-rgb.yml')
        with self.assertRaises(ServiceBadRequestError) as cm:
            get_dataset_tile(ctx, 'demo-rgb', 'rgb', '0', '0', '0', RequestParamsMock(b='refl_3'))
        self.assertEqual(400, cm.exception.status_code)
        self.assertEqual("Variable 'refl_3' not found in dataset 'demo-rgb'",
                         cm.exception.reason)

    def test_get_dataset_rgb_tile_no_vars(self):
        ctx = new_test_service_context()
        with self.assertRaises(ServiceBadRequestError) as cm:
            get_dataset_tile(ctx, 'demo', 'rgb', '0', '0', '0', RequestParamsMock())
        self.assertEqual(400, cm.exception.status_code)
        self.assertEqual("No variable in dataset 'demo' specified for RGB",
                         cm.exception.reason)

    def test_get_ne2_tile(self):
        ctx = new_test_service_context()
        tile = get_ne2_tile(ctx, '0', '0', '0', RequestParamsMock())
        self.assertIsInstance(tile, bytes)

    def test_get_dataset_tile_grid(self):
        self.maxDiff = None

        ctx = new_test_service_context()
        tile_grid = get_dataset_tile_grid(ctx, 'demo', 'conc_chl', 'ol4', 'http://bibo')
        self.assertEqual(
            {
                'maxZoom': 2,
                'minZoom': 0,
                'projection': 'EPSG:4326',
                'tileGrid': {'extent': [0, 50, 5, 52.5],
                             'origin': [0, 52.5],
                             'resolutions': [0.01, 0.005, 0.0025],
                             'tileSize': [250, 250]},
                'url': 'http://bibo/datasets/demo/vars/conc_chl/tiles/{z}/{x}/{y}.png'
            },
            tile_grid)

        tile_grid = get_dataset_tile_grid(ctx, 'demo', 'conc_chl', 'cesium', 'http://bibo')
        self.assertEqual({
            'url': self.base_url + '/datasets/demo/vars/conc_chl/tiles/{z}/{x}/{y}.png',
            'rectangle': dict(west=0.0, south=50.0, east=5.0, north=52.5),
            'minimumLevel': 0,
            'maximumLevel': 2,
            'tileWidth': 250,
            'tileHeight': 250,
            'tilingScheme': {'rectangle': dict(west=0.0, south=50.0, east=5.0, north=52.5),
                             'numberOfLevelZeroTilesX': 2,
                             'numberOfLevelZeroTilesY': 1},
        }, tile_grid)

        with self.assertRaises(ServiceBadRequestError) as cm:
            get_dataset_tile_grid(ctx, 'demo', 'conc_chl', 'ol2.json', 'http://bibo')
        self.assertEqual(400, cm.exception.status_code)
        self.assertEqual('Unknown tile client "ol2.json"', cm.exception.reason)

    def test_get_dataset_tile_grid_with_prefix(self):
        self.maxDiff = None

        ctx = new_test_service_context(prefix='api/v1')

        tile_grid = get_dataset_tile_grid(ctx, 'demo', 'conc_chl', 'ol4', 'http://bibo')
        self.assertEqual(
            {
                'maxZoom': 2,
                'minZoom': 0,
                'projection': 'EPSG:4326',
                'tileGrid': {'extent': [0, 50, 5, 52.5],
                             'origin': [0, 52.5],
                             'resolutions': [0.01, 0.005, 0.0025],
                             'tileSize': [250, 250]},
                'url': 'http://bibo/api/v1/datasets/demo/vars/conc_chl/tiles/{z}/{x}/{y}.png'
            },
            tile_grid)

    def test_get_legend(self):
        ctx = new_test_service_context()
        image = get_legend(ctx, 'demo', 'conc_chl', RequestParamsMock())
        self.assertEqual("<class 'bytes'>", str(type(image)))

        # This is fine, because we fall back to "viridis".
        image = get_legend(ctx, 'demo', 'conc_chl', RequestParamsMock(cbar='sun-shine'))
        self.assertEqual("<class 'bytes'>", str(type(image)))

        with self.assertRaises(ServiceBadRequestError) as cm:
            get_legend(ctx, 'demo', 'conc_chl', RequestParamsMock(vmin='sun-shine'))
        self.assertEqual("""Parameter "vmin" must be a number, but was 'sun-shine'""", cm.exception.reason)

        with self.assertRaises(ServiceBadRequestError) as cm:
            get_legend(ctx, 'demo', 'conc_chl', RequestParamsMock(width='sun-shine'))
        self.assertEqual("""Parameter "width" must be an integer, but was 'sun-shine'""", cm.exception.reason)

    def test_get_ne2_tile_grid(self):
        ctx = ServiceContext()
        tile_grid = get_ne2_tile_grid(ctx, 'ol4', 'http://bibo')
        self.assertEqual({
            'url': self.base_url + '/ne2/tiles/{z}/{x}/{y}.jpg',
            'projection': 'EPSG:4326',
            'minZoom': 0,
            'maxZoom': 2,
            'tileGrid': {'extent': [-180.0, -90.0, 180.0, 90.0],
                         'origin': [-180.0, 90.0],
                         'resolutions': [0.703125, 0.3515625, 0.17578125],
                         'tileSize': [256, 256]},
        }, tile_grid)

        with self.assertRaises(ServiceBadRequestError) as cm:
            get_ne2_tile_grid(ctx, 'cesium', 'http://bibo')
        self.assertEqual(400, cm.exception.status_code)
        self.assertEqual("Unknown tile client 'cesium'", cm.exception.reason)

    @property
    def base_url(self):
        return f'http://bibo'
