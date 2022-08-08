import unittest
from importlib import resources as resources

from lxml import etree
import xml.etree.ElementTree as ElementTree

from test.webapi.helpers import get_api_ctx
from test.webapi.ows import res as test_res
from xcube.webapi.ows.wcs import res
from xcube.webapi.ows.wcs.context import WcsContext

from xcube.webapi.ows.wcs.controllers import get_capabilities_xml
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
