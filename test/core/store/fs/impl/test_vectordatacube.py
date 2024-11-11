# Copyright (c) 2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.core.new import new_vector_data_cube

from xcube.core.store.datatype import VECTOR_DATA_CUBE_TYPE
from xcube.core.store.fs.impl.vectordatacube import VectorDataCubeZarrFsDataAccessor
from xcube.core.store.fs.impl.vectordatacube import VectorDataCubeNetcdfFsDataAccessor


class VectorDataCubeZarrFsDataAccessorTest(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.accessor = VectorDataCubeZarrFsDataAccessor()
        # self.vdc = new_vector_data_cube()

    def test_get_data_type(self):
        self.assertEqual(VECTOR_DATA_CUBE_TYPE, self.accessor.get_data_type())

    def test_get_format_id(self) -> str:
        self.assertEqual("zarr", self.accessor.get_format_id())

    def test_write_and_open_data(self):
        vdc = new_vector_data_cube()
        data_id = self.accessor.write_data(
            vdc, data_id="vdc_test.zarr", protocol="file", root="/test_data/"
        )
        self.assertIsNotNone(data_id)
