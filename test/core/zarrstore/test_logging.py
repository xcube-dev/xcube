# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

from zarr.storage import MemoryStore

from xcube.core.zarrstore import LoggingZarrStore


class LoggingZarrStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.zattrs_value = bytes()
        self.original_store = MemoryStore()
        self.original_store.update({'chl/.zattrs': self.zattrs_value})

    def test_read(self):
        logging_store = LoggingZarrStore(self.original_store)

        # noinspection PyUnresolvedReferences
        self.assertEqual(['.zattrs'],
                         logging_store.listdir('chl'))
        # noinspection PyUnresolvedReferences
        self.assertEqual(0,
                         logging_store.getsize('chl'))
        self.assertEqual({'chl/.zattrs'},
                         set(logging_store.keys()))
        self.assertEqual(['chl/.zattrs'],
                         list(iter(logging_store)))
        self.assertTrue('chl/.zattrs' in logging_store)
        self.assertEqual(1,
                         len(logging_store))
        self.assertEqual(self.zattrs_value,
                         logging_store.get('chl/.zattrs'))
        # assert original_store not changed
        self.assertEqual({'chl/.zattrs'},
                         set(self.original_store.keys()))

    def test_write(self):
        logging_store = LoggingZarrStore(self.original_store)

        zarray_value = bytes()
        logging_store['chl/.zarray'] = zarray_value
        self.assertEqual({'chl/.zattrs',
                          'chl/.zarray'},
                         set(self.original_store.keys()))
        del logging_store['chl/.zarray']
        self.assertEqual({'chl/.zattrs'},
                         set(self.original_store.keys()))
