# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.webapi.common.xml import Document
from xcube.webapi.common.xml import Element


class XmlTest(unittest.TestCase):
    def test_document(self):
        root = Element(
            "Address",
            attrs=dict(id="0294650"),
            elements=[
                Element("Street", text="Bibo Drive 2b"),
                Element("City", text="NYC"),
            ],
        )
        document = Document(root)
        self.assertEqual(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Address id="0294650">\n'
            "  <Street>Bibo Drive 2b</Street>\n"
            "  <City>NYC</City>\n"
            "</Address>\n",
            document.to_xml(indent=2),
        )
        r = document.add(Element("PostalCode", text="87234"))
        self.assertIs(r, document)
        self.assertEqual(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Address id="0294650">\n'
            "  <Street>Bibo Drive 2b</Street>\n"
            "  <City>NYC</City>\n"
            "  <PostalCode>87234</PostalCode>\n"
            "</Address>\n",
            document.to_xml(indent=2),
        )

    def test_element(self):
        element = Element(
            "Address",
            attrs=dict(id="0294650"),
            elements=[
                Element("Street", text="Bibo Drive 2b"),
                Element("City", text="NYC"),
            ],
        )
        self.assertEqual(
            '<Address id="0294650">\n'
            "  <Street>Bibo Drive 2b</Street>\n"
            "  <City>NYC</City>\n"
            "</Address>",
            element.to_xml(indent=2),
        )
        r = element.add(Element("PostalCode", text="87234"))
        self.assertIs(r, element)
        self.assertEqual(
            '<Address id="0294650">\n'
            "  <Street>Bibo Drive 2b</Street>\n"
            "  <City>NYC</City>\n"
            "  <PostalCode>87234</PostalCode>\n"
            "</Address>",
            element.to_xml(indent=2),
        )
