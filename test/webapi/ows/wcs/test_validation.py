import unittest

from lxml import etree

from xcube.webapi.ows.wcs import res
from test.webapi.ows import res as test_res
import importlib.resources as resources


class ValidationTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_validate_minimum(self):
        xml_text = resources.read_text(test_res, 'WCSCapabilities_minimum.xml')
        schema_text = resources.read_text(res, 'wcsCapabilities.xsd')

        xml = etree.fromstring(xml_text)
        xsd = etree.XMLSchema(etree.fromstring(schema_text))
        xsd.assertValid(xml)
