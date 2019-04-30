from tornado.testing import AsyncHTTPTestCase

from xcube.webapi.app import new_application
# For usage of the tornado.testing.AsyncHTTPTestCase see http://www.tornadoweb.org/en/stable/testing.html
from xcube.webapi.defaults import API_PREFIX, DEFAULT_NAME
from .helpers import new_test_service_context


class HandlersTest(AsyncHTTPTestCase):

    def get_app(self):
        application = new_application()
        application.service_context = new_test_service_context()
        return application

    def assertResponseOK(self, response):
        self.assertEqual(200, response.code, response.reason)
        self.assertEqual("OK", response.reason)

    def assertBadRequestResponse(self, response, expected_reason="Bad Request"):
        self.assertEqual(400, response.code)
        self.assertEqual(expected_reason, response.reason)

    def assertResourceNotFoundResponse(self, response, expected_reason="Not Found"):
        self.assertEqual(404, response.code)
        self.assertEqual(expected_reason, response.reason)

    def test_fetch_base(self):
        response = self.fetch(self.prefix + '/')
        self.assertResponseOK(response)

    def test_fetch_wmts_kvp_capabilities(self):
        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?SERVICE=WMTS'
                                            '&VERSION=1.0.0'
                                            '&REQUEST=GetCapabilities')
        self.assertResponseOK(response)

        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?service=WMTS'
                                            '&version=1.0.0'
                                            '&request=GetCapabilities')
        self.assertResponseOK(response)

        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?Service=WMTS'
                                            '&Version=1.0.0'
                                            '&Request=GetCapabilities')
        self.assertResponseOK(response)

        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?VERSION=1.0.0&REQUEST=GetCapabilities')
        self.assertBadRequestResponse(response, 'Missing query parameter "service"')

        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?SERVICE=WMS'
                                            'VERSION=1.0.0'
                                            '&REQUEST=GetCapabilities')
        self.assertBadRequestResponse(response, 'Value for "service" parameter must be "WMTS"')

    def test_fetch_wmts_kvp_tile(self):
        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?Service=WMTS'
                                            '&Version=1.0.0'
                                            '&Request=GetTile'
                                            '&Format=image/png'
                                            '&Style=Default'
                                            '&Layer=demo.conc_chl'
                                            '&TileMatrixSet=TileGrid_2000_1000'
                                            '&TileMatrix=0'
                                            '&TileRow=0'
                                            '&TileCol=0')
        self.assertResponseOK(response)

        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?Service=WMTS'
                                            '&Version=1.0.0'
                                            '&Request=GetTile'
                                            '&Format=image/jpg'
                                            '&Style=Default'
                                            '&Layer=demo.conc_chl'
                                            '&TileMatrixSet=TileGrid_2000_1000'
                                            '&TileMatrix=0'
                                            '&TileRow=0'
                                            '&TileCol=0')
        self.assertBadRequestResponse(response, 'Value for "format" parameter must be "image/png"')

        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?Service=WMTS'
                                            '&Version=1.1.0'
                                            '&Request=GetTile'
                                            '&Format=image/png'
                                            '&Style=Default'
                                            '&Layer=demo.conc_chl'
                                            '&TileMatrixSet=TileGrid_2000_1000'
                                            '&TileMatrix=0'
                                            '&TileRow=0'
                                            '&TileCol=0')
        self.assertBadRequestResponse(response, 'Value for "version" parameter must be "1.0.0"')

        response = self.fetch(self.prefix + '/wmts/kvp'
                                            '?Service=WMTS'
                                            '&Request=GetTile'
                                            '&Version=1.0.0'
                                            '&Format=image/png'
                                            '&Style=Default'
                                            '&Layer=conc_chl'
                                            '&TileMatrixSet=TileGrid_2000_1000'
                                            '&TileMatrix=0'
                                            '&TileRow=0'
                                            '&TileCol=0')
        self.assertBadRequestResponse(response, 'Value for "layer" parameter must be "<dataset>.<variable>"')

    def test_fetch_wmts_capabilities(self):
        response = self.fetch(self.prefix + '/wmts/1.0.0/WMTSCapabilities.xml')
        self.assertResponseOK(response)

    def test_fetch_wmts_tile(self):
        response = self.fetch(self.prefix + '/wmts/1.0.0/tile/demo/conc_chl/0/0/0.png')
        self.assertResponseOK(response)

    def test_fetch_wmts_tile_with_params(self):
        response = self.fetch(self.prefix + '/wmts/1.0.0/tile/demo/conc_chl/0/0/0.png?time=current&cbar=jet')
        self.assertResponseOK(response)

    def test_fetch_datasets_json(self):
        response = self.fetch(self.prefix + '/datasets')
        self.assertResponseOK(response)

    def test_fetch_datasets_details_json(self):
        response = self.fetch(self.prefix + '/datasets?details=1')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/datasets?details=1&tiles=cesium')
        self.assertResponseOK(response)

    def test_fetch_dataset_json(self):
        response = self.fetch(self.prefix + '/datasets/demo')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/datasets/demo?tiles=ol4')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/datasets/demo?tiles=cesium')
        self.assertResponseOK(response)

    def test_fetch_coords_json(self):
        response = self.fetch(self.prefix + '/datasets/demo/coords/time')
        self.assertResponseOK(response)

    def test_fetch_dataset_tile(self):
        response = self.fetch(self.prefix + '/datasets/demo/vars/conc_chl/tiles/0/0/0.png')
        self.assertResponseOK(response)

    def test_fetch_dataset_tile_with_params(self):
        response = self.fetch(
            self.prefix + '/datasets/demo/vars/conc_chl/tiles/0/0/0.png?time=current&cbar=jet&debug=1')
        self.assertResponseOK(response)

    def test_fetch_dataset_tile_grid_ol4_json(self):
        response = self.fetch(self.prefix + '/datasets/demo/vars/conc_chl/tilegrid?tiles=ol4')
        self.assertResponseOK(response)

    def test_fetch_dataset_tile_grid_cesium_json(self):
        response = self.fetch(self.prefix + '/datasets/demo/vars/conc_chl/tilegrid?tiles=cesium')
        self.assertResponseOK(response)

    def test_fetch_ne2_tile(self):
        response = self.fetch(self.prefix + '/ne2/tiles/0/0/0.jpg')
        self.assertResponseOK(response)

    def test_fetch_ne2_tile_grid(self):
        response = self.fetch(self.prefix + '/ne2/tilegrid?tiles=ol4')
        self.assertResponseOK(response)

    def test_fetch_color_bars_json(self):
        response = self.fetch(self.prefix + '/colorbars')
        self.assertResponseOK(response)

    def test_fetch_color_bars_html(self):
        response = self.fetch(self.prefix + '/colorbars.html')
        self.assertResponseOK(response)

    def test_fetch_feature_collections(self):
        response = self.fetch(self.prefix + '/places')
        self.assertResponseOK(response)

    def test_fetch_features(self):
        response = self.fetch(self.prefix + '/places/all')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/places/all?bbox=10,10,20,20')
        self.assertResponseOK(response)

    def test_fetch_features_for_dataset(self):
        response = self.fetch(self.prefix + '/places/all/demo')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/places/inside-cube/demo')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/places/bibo/demo')
        self.assertResourceNotFoundResponse(response, 'Place group "bibo" not found')
        response = self.fetch(self.prefix + '/places/inside-cube/bibo')
        self.assertResourceNotFoundResponse(response, 'Dataset "bibo" not found')

    def test_fetch_time_series_info(self):
        response = self.fetch(self.prefix + '/ts')
        self.assertResponseOK(response)

    def test_fetch_time_series_point(self):
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/point')
        self.assertBadRequestResponse(response, 'Missing query parameter "lon"')
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/point?lon=2.1')
        self.assertBadRequestResponse(response, 'Missing query parameter "lat"')
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/point?lon=120.5&lat=-12.4')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/point?lon=2.1&lat=51.1')
        self.assertResponseOK(response)

    def test_fetch_time_series_geometry(self):
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometry', method="POST",
                              body='')
        self.assertBadRequestResponse(response, 'Invalid or missing GeoJSON geometry in request body')
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometry', method="POST",
                              body='{"type":"Point"}')
        self.assertBadRequestResponse(response, 'Invalid GeoJSON geometry')
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometry', method="POST",
                              body='{"type": "Point", "coordinates": [1, 51]}')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometry', method="POST",
                              body='{"type":"Polygon", "coordinates": [[[1, 51], [2, 51], [2, 52], [1, 51]]]}')
        self.assertResponseOK(response)

    def test_fetch_time_series_geometries(self):
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometries', method="POST",
                              body='')
        self.assertBadRequestResponse(response, 'Invalid or missing GeoJSON geometry collection in request body')
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometries', method="POST",
                              body='{"type":"Point"}')
        self.assertBadRequestResponse(response, 'Invalid GeoJSON geometry collection')
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometries', method="POST",
                              body='{"type": "GeometryCollection", "geometries": null}')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometries', method="POST",
                              body='{"type": "GeometryCollection", "geometries": []}')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/geometries', method="POST",
                              body='{"type": "GeometryCollection", "geometries": [{"type": "Point", "coordinates": [1, 51]}]}')
        self.assertResponseOK(response)

    def test_fetch_time_series_features(self):
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/places', method="POST",
                              body='')
        self.assertBadRequestResponse(response, 'Invalid or missing GeoJSON feature collection in request body')
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/places', method="POST",
                              body='{"type":"Point"}')
        self.assertBadRequestResponse(response, 'Invalid GeoJSON feature collection')
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/places', method="POST",
                              body='{"type": "FeatureCollection", "features": null}')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/places', method="POST",
                              body='{"type": "FeatureCollection", "features": []}')
        self.assertResponseOK(response)
        response = self.fetch(self.prefix + '/ts/demo/conc_chl/places', method="POST",
                              body='{"type": "FeatureCollection", "features": ['
                                   '  {"type": "Feature", "properties": {}, '
                                   '   "geometry": {"type": "Point", "coordinates": [1, 51]}}'
                                   ']}')
        self.assertResponseOK(response)

    @property
    def prefix(self):
        return f"/{DEFAULT_NAME}{API_PREFIX}"
