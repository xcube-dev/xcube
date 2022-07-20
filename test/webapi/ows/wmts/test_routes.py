# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from ...helpers import RoutesTestCase


class WmtsRoutesTest(RoutesTestCase):

    def test_fetch_wmts_kvp_capabilities(self):
        response = self.fetch('/wmts/kvp'
                              '?SERVICE=WMTS'
                              '&VERSION=1.0.0'
                              '&REQUEST=GetCapabilities')
        self.assertResponseOK(response)

        response = self.fetch('/wmts/kvp'
                              '?service=WMTS'
                              '&version=1.0.0'
                              '&request=GetCapabilities')
        self.assertResponseOK(response)

        response = self.fetch('/wmts/kvp'
                              '?Service=WMTS'
                              '&Version=1.0.0'
                              '&Request=GetCapabilities')
        self.assertResponseOK(response)

        response = self.fetch('/wmts/kvp'
                              '?VERSION=1.0.0&REQUEST=GetCapabilities')
        self.assertBadRequestResponse(
            response,
            expected_message='value for "service" parameter must be "WMTS"'
        )

        response = self.fetch('/wmts/kvp'
                              '?SERVICE=WMS'
                              'VERSION=1.0.0'
                              '&REQUEST=GetCapabilities')
        self.assertBadRequestResponse(
            response,
            expected_message='value for "service" parameter must be "WMTS"'
        )

    def test_fetch_wmts_kvp_tile(self):
        response = self.fetch('/wmts/kvp'
                              '?Service=WMTS'
                              '&Version=1.0.0'
                              '&Request=GetTile'
                              '&Format=image/png'
                              '&Style=Default'
                              '&Layer=demo.conc_chl'
                              '&TileMatrixSet=WorldCRS84Quad'
                              '&TileMatrix=0'
                              '&TileRow=0'
                              '&TileCol=0')
        self.assertResponseOK(response)

        # issue #132 by Dirk
        response = self.fetch('/wmts/kvp'
                              '?Service=WMTS'
                              '&Version=1.0.0'
                              '&Request=GetTile'
                              '&Format=image/png'
                              '&Style=Default'
                              '&Layer=demo.conc_chl'
                              '&TileMatrixSet=WorldWebMercatorQuad'
                              '&TileMatrix=0'
                              '&TileRow=0'
                              '&TileCol=0'
                              '&Time=2017-01-25T09%3A35%3A50')
        self.assertResponseOK(response)

        # issue #132 by Dirk
        response = self.fetch('/wmts/kvp'
                              '?Service=WMTS'
                              '&Version=1.0.0'
                              '&Request=GetTile'
                              '&Format=image/png'
                              '&Style=Default'
                              '&Layer=demo.conc_chl'
                              '&TileMatrixSet=WorldWebMercatorQuad'
                              '&TileMatrix=0'
                              '&TileRow=0'
                              '&TileCol=0'
                              '&Time=2017-01-25T09%3A35%3A50%2F2017-01-25T10%3A20%3A15')
        self.assertResponseOK(response)

        response = self.fetch('/wmts/kvp'
                              '?Service=WMTS'
                              '&Version=1.0.0'
                              '&Request=GetTile'
                              '&Format=image/jpg'
                              '&Style=Default'
                              '&Layer=demo.conc_chl'
                              '&TileMatrixSet=WorldCRS84Quad'
                              '&TileMatrix=0'
                              '&TileRow=0'
                              '&TileCol=0')
        self.assertBadRequestResponse(
            response,
            'value for "format" parameter must be "image/png"'
        )

        response = self.fetch('/wmts/kvp'
                              '?Service=WMTS'
                              '&Version=1.1.0'
                              '&Request=GetTile'
                              '&Format=image/png'
                              '&Style=Default'
                              '&Layer=demo.conc_chl'
                              '&TileMatrixSet=WorldCRS84Quad'
                              '&TileMatrix=0'
                              '&TileRow=0'
                              '&TileCol=0')
        self.assertBadRequestResponse(
            response,
            'value for "version" parameter must be "1.0.0"'
        )

        response = self.fetch('/wmts/kvp'
                              '?Service=WMTS'
                              '&Request=GetTile'
                              '&Version=1.0.0'
                              '&Format=image/png'
                              '&Style=Default'
                              '&Layer=conc_chl'
                              '&TileMatrixSet=WorldCRS84Quad'
                              '&TileMatrix=0'
                              '&TileRow=0'
                              '&TileCol=0')
        self.assertBadRequestResponse(
            response,
            'value for "layer" parameter must be "<dataset>.<variable>"'
        )

        response = self.fetch('/wmts/kvp'
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
        self.assertBadRequestResponse(
            response,
            'value for "tilematrixset" parameter must'
            ' be one of (\'WorldCRS84Quad\', \'WorldWebMercatorQuad\')'
        )

    def test_fetch_wmts_capabilities(self):
        response = self.fetch(
            '/wmts/1.0.0/WMTSCapabilities.xml')
        self.assertResponseOK(response)

    def test_fetch_wmts_tile(self):
        response = self.fetch(
            '/wmts/1.0.0/tile/demo/conc_chl/0/0/0.png'
        )
        self.assertResponseOK(response)

    def test_fetch_wmts_tile_geo(self):
        response = self.fetch(
            '/wmts/1.0.0/tile/demo/conc_chl/WorldCRS84Quad/0/0/0.png'
        )
        self.assertResponseOK(response)

    def test_fetch_wmts_tile_mercator(self):
        response = self.fetch(
            '/wmts/1.0.0/tile/demo/conc_chl/WorldWebMercatorQuad/0/0/0.png'
        )
        self.assertResponseOK(response)

    def test_fetch_wmts_tile_with_params(self):
        response = self.fetch(
            '/wmts/1.0.0/tile/demo/conc_chl/0/0/0.png'
            '?time=current&cbar=jet'
        )
        self.assertResponseOK(response)

    def test_fetch_wmts_tile_with_params_geo(self):
        response = self.fetch(
            '/wmts/1.0.0/tile/demo/conc_chl/0/0/0.png'
            '?time=current&cbar=jet&TileMatrixSet=WorldCRS84Quad'
        )
        self.assertResponseOK(response)

    def test_fetch_wmts_tile_with_params_mercator(self):
        response = self.fetch(
            '/wmts/1.0.0/tile/demo/conc_chl/0/0/0.png'
            '?time=current&cbar=jet&TileMatrixSet=WorldWebMercatorQuad'
        )
        self.assertResponseOK(response)
