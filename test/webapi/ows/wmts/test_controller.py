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

import os
import unittest

import pyproj

from test.webapi.helpers import get_api_ctx
from test.webapi.helpers import get_server
from xcube.core.gridmapping import GridMapping
from xcube.core.tilingscheme import TilingScheme
from xcube.webapi.ows.wmts.context import WmtsContext
from xcube.webapi.ows.wmts.controllers import (
    get_operations_metadata_element,
    get_service_identification_element,
    get_service_provider_element,
    get_tile_matrix_set_crs84_element,
    get_tile_matrix_set_web_mercator_element,
    get_wmts_capabilities_xml,
    get_crs84_bbox,
    WMTS_CRS84_TMS_ID,
    WMTS_WEB_MERCATOR_TMS_ID
)


def get_test_res_path(path: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__),
                                         'res', path))


class WmtsControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.wmts_ctx = get_api_ctx('ows.wmts', WmtsContext)

    def test_get_wmts_capabilities_xml_crs84(self):
        self.maxDiff = None
        with open(get_test_res_path('WMTSCapabilities-CRS84.xml')) as fp:
            expected_xml = fp.read()
        actual_xml = get_wmts_capabilities_xml(self.wmts_ctx,
                                               'http://bibo',
                                               tms_id=WMTS_CRS84_TMS_ID)
        # Do not delete, useful for debugging
        # print(80 * '=')
        # print(actual_xml)
        # print(80 * '=')
        self.assertEqual(expected_xml, actual_xml)

    def test_get_wmts_capabilities_xml_web_mercator(self):
        self.maxDiff = None
        with open(get_test_res_path('WMTSCapabilities-OSM.xml')) as fp:
            expected_xml = fp.read()
        actual_xml = get_wmts_capabilities_xml(self.wmts_ctx,
                                               'http://bibo',
                                               WMTS_WEB_MERCATOR_TMS_ID)
        # Do not delete, useful for debugging
        # print(80 * '=')
        # print(actual_xml)
        # print(80 * '=')
        self.assertEqual(expected_xml, actual_xml)


class WmtsControllerXmlGenTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        server = get_server()
        config = dict(server.ctx.config)
        config.update(
            ServiceProvider=dict(
                ProviderName='Bibo',
                ServiceContact=dict(
                    PositionName='Boss',
                    ContactInfo=dict(
                        Address=dict(
                            City='NYC'
                        ),
                    )
                ),
            )
        )
        server.update(config)
        self.wmts_ctx = server.ctx.get_api_ctx('ows.wmts', cls=WmtsContext)

    def test_get_service_provider(self):
        element = get_service_provider_element(self.wmts_ctx)
        self.assertEqual(
            '<ows:ServiceProvider>\n'
            '  <ows:ProviderName>Bibo</ows:ProviderName>\n'
            '  <ows:ProviderSite xlink:href=""/>\n'
            '  <ows:ServiceContact>\n'
            '    <ows:IndividualName/>\n'
            '    <ows:PositionName>Boss</ows:PositionName>\n'
            '    <ows:ContactInfo>\n'
            '      <ows:Phone>\n'
            '        <ows:Voice/>\n'
            '        <ows:Facsimile/>\n'
            '      </ows:Phone>\n'
            '      <ows:Address>\n'
            '        <ows:DeliveryPoint/>\n'
            '        <ows:City>NYC</ows:City>\n'
            '        <ows:AdministrativeArea/>\n'
            '        <ows:PostalCode/>\n'
            '        <ows:Country/>\n'
            '        <ows:ElectronicMailAddress/>\n'
            '      </ows:Address>\n'
            '    </ows:ContactInfo>\n'
            '  </ows:ServiceContact>\n'
            '</ows:ServiceProvider>',
            element.to_xml(indent=2)
        )

    def test_get_operations_metadata(self):
        element = get_operations_metadata_element(self.wmts_ctx,
                                                  'https://bibo',
                                                  WMTS_WEB_MERCATOR_TMS_ID)
        self.assertEqual(
            '<ows:OperationsMetadata>\n'
            '  <ows:Operation name="GetCapabilities">\n'
            '    <ows:DCP>\n'
            '      <ows:HTTP>\n'
            '        <ows:Get xlink:href="https://bibo/wmts/kvp?">\n'
            '          <ows:Constraint name="GetEncoding">\n'
            '            <ows:AllowedValues>\n'
            '              <ows:Value>KVP</ows:Value>\n'
            '            </ows:AllowedValues>\n'
            '          </ows:Constraint>\n'
            '        </ows:Get>\n'
            '        <ows:Get xlink:href="https://bibo/wmts/'
            '1.0.0/WorldWebMercatorQuad/WMTSCapabilities.xml">\n'
            '          <ows:Constraint name="GetEncoding">\n'
            '            <ows:AllowedValues>\n'
            '              <ows:Value>REST</ows:Value>\n'
            '            </ows:AllowedValues>\n'
            '          </ows:Constraint>\n'
            '        </ows:Get>\n'
            '      </ows:HTTP>\n'
            '    </ows:DCP>\n'
            '  </ows:Operation>\n'
            '  <ows:Operation name="GetTile">\n'
            '    <ows:DCP>\n'
            '      <ows:HTTP>\n'
            '        <ows:Get xlink:href="https://bibo/wmts/kvp?">\n'
            '          <ows:Constraint name="GetEncoding">\n'
            '            <ows:AllowedValues>\n'
            '              <ows:Value>KVP</ows:Value>\n'
            '            </ows:AllowedValues>\n'
            '          </ows:Constraint>\n'
            '        </ows:Get>\n'
            '        <ows:Get xlink:href="https://bibo/wmts/1.0.0/">\n'
            '          <ows:Constraint name="GetEncoding">\n'
            '            <ows:AllowedValues>\n'
            '              <ows:Value>REST</ows:Value>\n'
            '            </ows:AllowedValues>\n'
            '          </ows:Constraint>\n'
            '        </ows:Get>\n'
            '      </ows:HTTP>\n'
            '    </ows:DCP>\n'
            '  </ows:Operation>\n'
            '</ows:OperationsMetadata>',
            element.to_xml(indent=2)
        )

    def test_get_tile_matrix_set_crs84_element(self):
        tiling_scheme = TilingScheme.GEOGRAPHIC.derive(
            min_level=0,
            max_level=2
        )
        element = get_tile_matrix_set_crs84_element(tiling_scheme)
        self.assertEqual(
            '<TileMatrixSet>\n'
            '  <ows:Identifier>WorldCRS84Quad</ows:Identifier>\n'
            '  <ows:Title>CRS84 for the World</ows:Title>\n'
            '  <ows:SupportedCRS>urn:ogc:def:crs:OGC:1.3:CRS84'
            '</ows:SupportedCRS>\n'
            '  <ows:BoundingBox crs="urn:ogc:def:crs:OGC:1.3:CRS84">\n'
            '    <ows:LowerCorner>-180 -90</ows:LowerCorner>\n'
            '    <ows:UpperCorner>180 90</ows:UpperCorner>\n'
            '  </ows:BoundingBox>\n'
            '  <WellKnownScaleSet>urn:ogc:def:wkss:OGC:1.0:GoogleCRS84Quad'
            '</WellKnownScaleSet>\n'
            '  <TileMatrix>\n'
            '    <ows:Identifier>0</ows:Identifier>\n'
            '    <ScaleDenominator>279541132.01435894</ScaleDenominator>\n'
            '    <TopLeftCorner>-180 90</TopLeftCorner>\n'
            '    <TileWidth>256</TileWidth>\n'
            '    <TileHeight>256</TileHeight>\n'
            '    <MatrixWidth>2</MatrixWidth>\n'
            '    <MatrixHeight>1</MatrixHeight>\n'
            '  </TileMatrix>\n'
            '  <TileMatrix>\n'
            '    <ows:Identifier>1</ows:Identifier>\n'
            '    <ScaleDenominator>139770566.00717947</ScaleDenominator>\n'
            '    <TopLeftCorner>-180 90</TopLeftCorner>\n'
            '    <TileWidth>256</TileWidth>\n'
            '    <TileHeight>256</TileHeight>\n'
            '    <MatrixWidth>4</MatrixWidth>\n'
            '    <MatrixHeight>2</MatrixHeight>\n'
            '  </TileMatrix>\n'
            '  <TileMatrix>\n'
            '    <ows:Identifier>2</ows:Identifier>\n'
            '    <ScaleDenominator>69885283.00358973</ScaleDenominator>\n'
            '    <TopLeftCorner>-180 90</TopLeftCorner>\n'
            '    <TileWidth>256</TileWidth>\n'
            '    <TileHeight>256</TileHeight>\n'
            '    <MatrixWidth>8</MatrixWidth>\n'
            '    <MatrixHeight>4</MatrixHeight>\n'
            '  </TileMatrix>\n'
            '</TileMatrixSet>',
            element.to_xml(indent=2)
        )

    def test_get_tile_matrix_set_web_mercator_element(self):
        tiling_scheme = TilingScheme.WEB_MERCATOR.derive(
            min_level=0,
            max_level=2
        )
        element = get_tile_matrix_set_web_mercator_element(tiling_scheme)
        self.assertEqual(
            ('<TileMatrixSet>\n'
             '  <ows:Identifier>WorldWebMercatorQuad</ows:Identifier>\n'
             '  <ows:Title>Google Maps Compatible for the World</ows:Title>\n'
             '  <ows:SupportedCRS>'
             'urn:ogc:def:crs:EPSG::3857</ows:SupportedCRS>\n'
             '  <ows:BoundingBox crs="urn:ogc:def:crs:EPSG::3857">\n'
             '    <ows:LowerCorner>'
             '-20037508.3427892 -20037508.3427892</ows:LowerCorner>\n'
             '    <ows:UpperCorner>'
             '20037508.3427892 20037508.3427892</ows:UpperCorner>\n'
             '  </ows:BoundingBox>\n'
             '  '
             '<WellKnownScaleSet>'
             'urn:ogc:def:wkss:OGC:1.0:GoogleMapsCompatible'
             '</WellKnownScaleSet>\n'
             '  <TileMatrix>\n'
             '    <ows:Identifier>0</ows:Identifier>\n'
             '    <ScaleDenominator>559082264.0287178</ScaleDenominator>\n'
             '    <TopLeftCorner>'
             '-20037508.3427892 20037508.3427892</TopLeftCorner>\n'
             '    <TileWidth>256</TileWidth>\n'
             '    <TileHeight>256</TileHeight>\n'
             '    <MatrixWidth>1</MatrixWidth>\n'
             '    <MatrixHeight>1</MatrixHeight>\n'
             '  </TileMatrix>\n'
             '  <TileMatrix>\n'
             '    <ows:Identifier>1</ows:Identifier>\n'
             '    <ScaleDenominator>279541132.0143589</ScaleDenominator>\n'
             '    <TopLeftCorner>'
             '-20037508.3427892 20037508.3427892</TopLeftCorner>\n'
             '    <TileWidth>256</TileWidth>\n'
             '    <TileHeight>256</TileHeight>\n'
             '    <MatrixWidth>2</MatrixWidth>\n'
             '    <MatrixHeight>2</MatrixHeight>\n'
             '  </TileMatrix>\n'
             '  <TileMatrix>\n'
             '    <ows:Identifier>2</ows:Identifier>\n'
             '    <ScaleDenominator>139770566.00717944</ScaleDenominator>\n'
             '    <TopLeftCorner>'
             '-20037508.3427892 20037508.3427892</TopLeftCorner>\n'
             '    <TileWidth>256</TileWidth>\n'
             '    <TileHeight>256</TileHeight>\n'
             '    <MatrixWidth>4</MatrixWidth>\n'
             '    <MatrixHeight>4</MatrixHeight>\n'
             '  </TileMatrix>\n'
             '</TileMatrixSet>'),
            element.to_xml(indent=2)
        )

    def test_service_identification_element(self):
        element = get_service_identification_element()
        self.assertEqual(
            (
                '<ows:ServiceIdentification>\n'
                '  <ows:Title>xcube WMTS</ows:Title>\n'
                '  <ows:Abstract>Web Map Tile Service (WMTS)'
                ' for xcube-conformant data cubes</ows:Abstract>\n'
                '  <ows:Keywords>\n'
                '    <ows:Keyword>tile</ows:Keyword>\n'
                '    <ows:Keyword>tile matrix set</ows:Keyword>\n'
                '    <ows:Keyword>map</ows:Keyword>\n'
                '  </ows:Keywords>\n'
                '  <ows:ServiceType>OGC WMTS</ows:ServiceType>\n'
                '  <ows:ServiceTypeVersion>1.0.0</ows:ServiceTypeVersion>\n'
                '  <ows:Fees>none</ows:Fees>\n'
                '  <ows:AccessConstraints>none</ows:AccessConstraints>\n'
                '</ows:ServiceIdentification>'
            ),
            element.to_xml(indent=2)
        )


class WmtsCrs84BboxTest(unittest.TestCase):
    def test_get_crs84_bbox_ok(self):
        t = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3035',
                                        always_xy=True)
        p0 = t.transform(10, 50)

        gm = GridMapping.regular((100, 100),
                                 p0,
                                 (1000., 1000.),
                                 pyproj.CRS.from_string('EPSG:3035'))
        bbox = get_crs84_bbox(gm)
        self.assertIsInstance(bbox, tuple)
        self.assertEqual(4, len(bbox))
        self.assertAlmostEqual(10.0, bbox[0])
        self.assertAlmostEqual(49.991462404901604, bbox[1])
        self.assertAlmostEqual(11.421266448415976, bbox[2])
        self.assertAlmostEqual(50.8990377520989, bbox[3])

    def test_get_crs84_bbox_fail(self):
        # construct impossible GridMapping
        gm = GridMapping.regular((100, 100),
                                 (100000000, -1000000),
                                 (1000., 1000.),
                                 pyproj.CRS.from_string('EPSG:3035'))
        with self.assertRaises(ValueError):
            get_crs84_bbox(gm)
