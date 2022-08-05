import unittest
from importlib import resources as resources

from lxml import etree
import xml.etree.ElementTree as ElementTree

from test.webapi.helpers import get_api_ctx
from test.webapi.ows import res as test_res
from xcube.webapi.ows.wcs import res
from xcube.webapi.ows.wcs.context import WcsContext

from xcube.webapi.ows.wcs.controllers import get_wcs_capabilities_xml


class ControllerTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.wcs_ctx = get_api_ctx('ows.wcs', WcsContext)

    # noinspection PyMethodMayBeStatic
    def test_get_capabilities(self):
        expected_xml = resources.read_text(test_res, 'WCSCapabilities.xml')
        actual_xml = get_wcs_capabilities_xml(
            self.wcs_ctx, 'https://xcube.brockmann-consult.de/wcs/kvp'
        )

        self.maxDiff = None
        # Do not delete, useful for debugging
        print(80 * '=')
        print(actual_xml)
        print(80 * '=')

        actual_xml = self.strip_whitespace(actual_xml)
        expected_xml = self.strip_whitespace(expected_xml)

        self.assertEqual(expected_xml, actual_xml)

    @staticmethod
    def strip_whitespace(xml: str) -> str:
        root = ElementTree.fromstring(xml)
        return ElementTree.canonicalize(ElementTree.tostring(root),
                                        strip_text=True)

    def test_how_to_validate(self):
        xml_text = resources.read_text(test_res, 'WCSCapabilities.xml')
        xml = etree.fromstring(xml_text)

        schema_text = resources.read_text(res, 'wcsCapabilities.xsd')
        xsd = etree.XMLSchema(etree.fromstring(schema_text))
        xsd.assertValid(xml)
