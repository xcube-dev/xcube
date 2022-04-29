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

from xcube.util.extend import extend


class ExtendTest(unittest.TestCase):
    def test_decorator(self):
        class MyBase:
            pass

        @extend(MyBase, "my_ext")
        class MyExt:
            """My pretty extension."""

            def __init__(self, base: MyBase):
                self._base = base

        self.assertTrue(hasattr(MyBase, "my_ext"))
        # noinspection PyUnresolvedReferences
        self.assertEqual("My pretty extension.", MyBase.my_ext.__doc__)

        my_base = MyBase()
        self.assertTrue(hasattr(my_base, "my_ext"))
        # noinspection PyUnresolvedReferences
        my_ext = my_base.my_ext
        self.assertIsInstance(my_ext, MyExt)
        self.assertEqual("My pretty extension.", my_ext.__doc__)
        self.assertIsInstance(my_ext._base, MyBase)

    def test_class_handler(self):
        class MyBase:
            extensions = []

            @classmethod
            def add_extension(cls, ext_item):
                MyBase.extensions.append(ext_item)

        @extend(MyBase, 'my_ext_1', class_handler='add_extension')
        class MyExt1:
            def __init__(self, base: MyBase):
                pass

        @extend(MyBase, 'my_ext_2', class_handler='add_extension')
        class MyExt2:
            def __init__(self, base: MyBase):
                pass

        self.assertEqual([('my_ext_1', MyExt1),
                          ('my_ext_2', MyExt2)],
                         MyBase.extensions)

    def test_instance_handler(self):
        class MyBase:
            def __init__(self):
                self.extensions = []

            def add_extension(self, ext_item):
                self.extensions.append(ext_item)

        @extend(MyBase, 'my_ext_1', inst_handler='add_extension')
        class MyExt1:
            def __init__(self, base: MyBase):
                pass

        @extend(MyBase, 'my_ext_2', inst_handler='add_extension')
        class MyExt2:
            def __init__(self, base: MyBase):
                pass

        my_base = MyBase()
        self.assertEqual([],
                         my_base.extensions)
        self.assertIsInstance(my_base.my_ext_1, MyExt1)
        self.assertIsInstance(my_base.my_ext_2, MyExt2)
        self.assertEqual(2,
                         len(my_base.extensions))

    def test_overwrite_docstring(self):
        class MyBase:
            pass

        class MyExt:
            """My pretty extension."""

            def __init__(self, base: MyBase):
                self._base = base

        extend(MyBase, "my_ext", doc="A better doc.")(MyExt)

        # noinspection PyUnresolvedReferences
        self.assertEqual("A better doc.", MyBase.my_ext.__doc__)

    def test_property_exists(self):
        class MyBase:
            @property
            def my_ext(self):
                return 42

        class MyExt:
            pass

        with self.assertRaises(ValueError) as cm:
            extend(MyBase, "my_ext")(MyExt)
        self.assertEqual('a property named my_ext'
                         ' already exists in class MyBase',
                         f'{cm.exception}')

    def test_illegal_property_name(self):
        class MyBase:
            pass

        class MyExt:
            pass

        with self.assertRaises(ValueError) as cm:
            extend(MyBase, "my ext")(MyExt)
        self.assertEqual("name must be a valid identifier,"
                         " but was 'my ext'",
                         f'{cm.exception}')

    def test_illegal_base_class(self):
        class MyExt:
            pass

        with self.assertRaises(ValueError) as cm:
            # noinspection PyTypeChecker
            extend("base_class", "my_ext")(MyExt)
        self.assertEqual('base_class must be a class type,'
                         ' but was str',
                         f'{cm.exception}')

    def test_illegal_ext_class(self):
        class MyBase:
            pass

        with self.assertRaises(ValueError) as cm:
            # noinspection PyTypeChecker
            extend(MyBase, "my_ext")("ext_class")
        self.assertEqual('the extend() decorator'
                         ' can be used with classes only',
                         f'{cm.exception}')
