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

import unittest

from xcube.webapi.common.xml import Document
from xcube.webapi.common.xml import Element


class XmlTest(unittest.TestCase):

    def test_document(self):
        root = Element('Address',
                       attrs=dict(id='0294650'),
                       elements=[
                           Element('Street', text='Bibo Drive 2b'),
                           Element('City', text='NYC'),
                       ])
        document = Document(root)
        self.assertEqual(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Address id="0294650">\n'
            '  <Street>Bibo Drive 2b</Street>\n'
            '  <City>NYC</City>\n'
            '</Address>\n',
            document.to_xml(indent=2)
        )
        r = document.add(Element('PostalCode', text='87234'))
        self.assertIs(r, document)
        self.assertEqual(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Address id="0294650">\n'
            '  <Street>Bibo Drive 2b</Street>\n'
            '  <City>NYC</City>\n'
            '  <PostalCode>87234</PostalCode>\n'
            '</Address>\n',
            document.to_xml(indent=2)
        )

    def test_element(self):
        element = Element('Address',
                          attrs=dict(id='0294650'),
                          elements=[
                              Element('Street', text='Bibo Drive 2b'),
                              Element('City', text='NYC'),
                          ])
        self.assertEqual(
            '<Address id="0294650">\n'
            '  <Street>Bibo Drive 2b</Street>\n'
            '  <City>NYC</City>\n'
            '</Address>',
            element.to_xml(indent=2)
        )
        r = element.add(Element('PostalCode', text='87234'))
        self.assertIs(r, element)
        self.assertEqual(
            '<Address id="0294650">\n'
            '  <Street>Bibo Drive 2b</Street>\n'
            '  <City>NYC</City>\n'
            '  <PostalCode>87234</PostalCode>\n'
            '</Address>',
            element.to_xml(indent=2)
        )
