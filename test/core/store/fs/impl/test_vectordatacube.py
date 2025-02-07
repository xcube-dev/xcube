# Copyright (c) 2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.core.new import new_vector_data_cube

from xcube.core.mldataset.abc import VectorDataCube
from xcube.core.store import new_data_store
from xcube.core.store.datatype import VECTOR_DATA_CUBE_TYPE
from xcube.core.store.fs.impl.vectordatacube import VectorDataCubeZarrFsDataAccessor


class VectorDataCubeStoreTest(unittest.TestCase):

    def test_write_to_and_read_from_store(self):
        store = new_data_store("file")
        vdc = new_vector_data_cube(
            variables=dict(
                precipitation=0.5,
                soilmoisture=1.0
            )
        )
        data_id = store.write_data(
            vdc, data_id="vdc_test.zarr", writer_id="vectordatacube:zarr:file"
        )
        self.assertIsNotNone(data_id)
        ds = store.open_data(data_id, opener_id="vectordatacube:zarr:file")
        self.assertIsNotNone(ds)
        self.assertIsInstance(ds, VectorDataCube)
        store.delete_data(data_id)

class VectorDataCubeZarrFsDataAccessorTest(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.accessor = VectorDataCubeZarrFsDataAccessor()

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
