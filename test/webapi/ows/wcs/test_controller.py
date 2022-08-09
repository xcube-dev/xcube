import unittest
import xml.etree.ElementTree as ElementTree
from importlib import resources as resources

from lxml import etree

from test.webapi.helpers import get_api_ctx
from test.webapi.ows import res as test_res
from xcube.webapi.ows.wcs import res
from xcube.webapi.ows.wcs.context import WcsContext
from xcube.webapi.ows.wcs.controllers import get_capabilities_xml, \
    validate_coverage_req
from xcube.webapi.ows.wcs.controllers import get_describe_xml


# noinspection PyMethodMayBeStatic
class ControllerTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.wcs_ctx = get_api_ctx('ows.wcs', WcsContext)

    def test_get_capabilities(self):
        actual_xml = get_capabilities_xml(
            self.wcs_ctx, 'https://xcube.brockmann-consult.de/wcs/kvp'
        )

        self.check_xml(actual_xml, 'WCSCapabilities.xml',
                       'wcsCapabilities.xsd')

    def test_describe_coverage(self):
        actual_xml = get_describe_xml(self.wcs_ctx)
        self.check_xml(actual_xml, 'WCSDescribe.xml', 'wcsDescribe.xsd')

    def test_describe_coverage_subset(self):
        actual_xml = get_describe_xml(self.wcs_ctx,
                                      [
                                          'demo-1w.quality_flags_stdev',
                                          'demo-1w.kd489_stdev',
                                          'demo.conc_chl'
                                      ])
        self.check_xml(actual_xml, 'WCSDescribe_subset.xml', 'wcsDescribe.xsd')

    def test_validate_coverage_request(self):
        # request is fine
        validate_coverage_req({
            'COVERAGE': 'demo.conc_chl',
            'CRS': 'EPSG:4326',
            'BBOX': '1 51 4 52',
            'WIDTH': 200,
            'HEIGHT': 200,
            'FORMAT': 'zarr'
        }, self.wcs_ctx)

        # TIME given in addition to BBOX -> fine
        validate_coverage_req({
            'COVERAGE': 'demo.conc_chl',
            'CRS': 'EPSG:4326',
            'BBOX': '1 51 4 52',
            'TIME': '2017-01-28 20:23:55.123456',
            'WIDTH': 200,
            'HEIGHT': 200,
            'FORMAT': 'zarr'
        }, self.wcs_ctx)

        # COVERAGE is missing -> expect a failure
        try:
            validate_coverage_req({
                'CRS': 'EPSG:4326',
                'BBOX': '1 51 4 52',
                'WIDTH': 200,
                'HEIGHT': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('No valid value for parameter COVERAGE provided. '
                             'COVERAGE must be a variable name prefixed with '
                             'its dataset name. Example: my_dataset.my_var',
                             str(e))

        # COVERAGE is given but not found -> expect a failure
        try:
            validate_coverage_req({
                'COVERAGE': 'invalid_coverage!',
                'CRS': 'EPSG:4326',
                'BBOX': '1 51 4 52',
                'WIDTH': 200,
                'HEIGHT': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('No valid value for parameter COVERAGE provided. '
                             'COVERAGE must be a variable name prefixed with '
                             'its dataset name. Example: my_dataset.my_var',
                             str(e))

        # TIME is used instead of BBOX -> fine
        validate_coverage_req({
            'COVERAGE': 'demo.conc_chl',
            'CRS': 'EPSG:4326',
            'TIME': '2020-01-28',
            'WIDTH': 200,
            'HEIGHT': 200,
            'FORMAT': 'zarr'
        }, self.wcs_ctx)

        # use invalid TIME format -> expect a failure
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'CRS': 'EPSG:4326',
                'TIME': '20201208',
                'WIDTH': 200,
                'HEIGHT': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('TIME value must be given in the format'
                             '\'YYYY-MM-DD[*HH[:MM[:SS[.mmm[mmm]]]]'
                             '[+HH:MM[:SS[.ffffff]]]]\'',
                             str(e))

        # RESX and RESY are given instead of W/H -> fine
        validate_coverage_req({
            'COVERAGE': 'demo.conc_chl',
            'CRS': 'EPSG:4326',
            'TIME': '2020-01-28',
            'RESX': 23.56,
            'RESY': 23.56,
            'FORMAT': 'zarr'
        }, self.wcs_ctx)

        # PARAMETER is given -> expect a failure (not yet supported)
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'PARAMETER': 'I expect nothing from using this parameter',
                'CRS': 'EPSG:4326',
                'TIME': '2020-12-08',
                'WIDTH': 200,
                'HEIGHT': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('PARAMETER not yet supported', str(e))

        # BBOX is given in wrong format -> expect a failure
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'CRS': 'EPSG:4326',
                'BBOX': '-10,3,-5,4',
                'WIDTH': 200,
                'HEIGHT': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('BBOX must be given as `minx miny maxx maxy`',
                             str(e))

        # WIDTH, but not HEIGHT is given -> expect a failure
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'CRS': 'EPSG:4326',
                'BBOX': '-10 3 -5 4',
                'WIDTH': 200,
                'RESY': 156.45,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('Either both WIDTH and HEIGHT, or both RESX and '
                             'RESY must be provided.', str(e))

        # HEIGHT, but not WIDTH is given -> expect a failure
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'CRS': 'EPSG:4326',
                'BBOX': '-10 3 -5 4',
                'HEIGHT': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('Either both WIDTH and HEIGHT, or both RESX and '
                             'RESY must be provided.', str(e))

        # WIDTH and HEIGHT and RESX and RESY are given -> expect a failure
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'CRS': 'EPSG:4326',
                'BBOX': '-10 3 -5 4',
                'WIDTH': 200,
                'HEIGHT': 200,
                'RESX': 200,
                'RESY': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('Either both WIDTH and HEIGHT, or both RESX and '
                             'RESY must be provided.', str(e))

        # INTERPOLATION is given -> expect a failure (not yet supported)
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'INTERPOLATION': 'Farest Neighbor',
                'CRS': 'EPSG:4326',
                'TIME': '2020-12-08',
                'WIDTH': 200,
                'HEIGHT': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('INTERPOLATION not yet supported', str(e))

        # EXCEPTIONS is given -> expect a failure (not yet supported)
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'EXCEPTIONS': 'Farest Neighbor',
                'CRS': 'EPSG:4326',
                'TIME': '2020-12-08',
                'WIDTH': 200,
                'HEIGHT': 200,
                'FORMAT': 'zarr'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('EXCEPTIONS not yet supported', str(e))

        # FORMAT is missing -> expect a failure
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'CRS': 'EPSG:4326',
                'TIME': '2020-12-08',
                'WIDTH': 200,
                'HEIGHT': 200,
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('FORMAT wrong or missing. Must be one of zarr, '
                             'netcdf4, csv', str(e))

        # FORMAT is invalid -> expect a failure
        try:
            validate_coverage_req({
                'COVERAGE': 'demo.conc_chl',
                'CRS': 'EPSG:4326',
                'TIME': '2020-12-08',
                'WIDTH': 200,
                'HEIGHT': 200,
                'FORMAT': 'MettCDF'
            }, self.wcs_ctx)
            self.fail('Classified invalid request as valid.')
        except ValueError as e:
            self.assertEqual('FORMAT wrong or missing. Must be one of zarr, '
                             'netcdf4, csv', str(e))

    def check_xml(self, actual_xml, expected_xml_resource, xsd):
        self.maxDiff = None
        # Do not delete, useful for debugging
        print(80 * '=')
        print(actual_xml)
        print(80 * '=')

        expected_xml = resources.read_text(test_res, expected_xml_resource)

        actual_xml = self.strip_whitespace(actual_xml)
        expected_xml = self.strip_whitespace(expected_xml)

        self.assertTrue(self.is_schema_compliant(
            expected_xml_resource, xsd)
        )
        self.assertEqual(expected_xml, actual_xml)

    @staticmethod
    def strip_whitespace(xml: str) -> str:
        root = ElementTree.fromstring(xml)
        return ElementTree.canonicalize(ElementTree.tostring(root),
                                        strip_text=True)

    def is_schema_compliant(self,
                            xml_resource,
                            xsd_resource):
        xml_text = resources.read_text(test_res, xml_resource)
        xml = etree.fromstring(xml_text)

        schema_text = resources.read_text(res, xsd_resource)
        xsd = etree.XMLSchema(etree.fromstring(schema_text))
        xsd.assertValid(xml)  # fail if xml is invalid
        return True
