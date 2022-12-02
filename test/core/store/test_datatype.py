import copy
import unittest

import geopandas
import xarray

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store.datatype import ANY_TYPE, DataTypeLike
from xcube.core.store.datatype import DATASET_TYPE
from xcube.core.store.datatype import DataType
from xcube.core.store.datatype import GEO_DATA_FRAME_TYPE
from xcube.core.store.datatype import MULTI_LEVEL_DATASET_TYPE
from xcube.util.jsonschema import JsonStringSchema


class A:
    pass


class B(A):
    pass


class C:
    pass


class DataTypeTest(unittest.TestCase):
    def test_normalize_to_any(self):
        self.assertNormalizeOk('any',
                               ANY_TYPE,
                               object,
                               'any',
                               ('any', '*', 'object', 'builtins.object'))
        self.assertNormalizeOk(None,
                               ANY_TYPE,
                               object,
                               'any',
                               ('any', '*', 'object', 'builtins.object'))
        self.assertNormalizeOk(type(None),
                               ANY_TYPE,
                               object,
                               'any',
                               ('any', '*', 'object', 'builtins.object'))

    def test_normalize_to_dataset(self):
        self.assertNormalizeOk('dataset',
                               DATASET_TYPE,
                               xarray.Dataset,
                               'dataset',
                               ('dataset',
                                'xarray.Dataset',
                                'xarray.core.dataset.Dataset'))

    def test_normalize_to_mldataset(self):
        self.assertNormalizeOk('mldataset',
                               MULTI_LEVEL_DATASET_TYPE,
                               MultiLevelDataset,
                               'mldataset',
                               ('mldataset',
                                'xcube.MultiLevelDataset',
                                'xcube.core.mldataset.MultiLevelDataset',
                                'xcube.core.mldataset.abc.MultiLevelDataset'))

    def test_normalize_to_geodataframe(self):
        self.assertNormalizeOk('geodataframe',
                               GEO_DATA_FRAME_TYPE,
                               geopandas.GeoDataFrame,
                               'geodataframe',
                               ('geodataframe',
                                'geopandas.GeoDataFrame',
                                'geopandas.geodataframe.GeoDataFrame'))

    def assertNormalizeOk(self,
                          data_type: DataTypeLike,
                          expected_data_type,
                          expected_dtype,
                          expected_alias,
                          expected_aliases):
        data_type = DataType.normalize(data_type)
        self.assertIs(expected_data_type, data_type)
        self.assertIs(expected_dtype, data_type.dtype)
        self.assertEqual(expected_alias, data_type.alias)
        self.assertEqual(expected_alias, str(data_type))
        self.assertEqual(f'{expected_alias!r}', repr(data_type))
        self.assertEqual(expected_aliases, data_type.aliases)
        for other_alias in data_type.aliases:
            self.assertIs(data_type,
                          DataType.normalize(other_alias))
        self.assertIs(expected_data_type,
                      DataType.normalize(expected_dtype))
        self.assertIs(expected_data_type,
                      DataType.normalize(expected_data_type))

    def test_normalize_non_default_types(self):
        data_type = DataType.normalize(str)
        self.assertIs(str, data_type.dtype)
        self.assertEqual('builtins.str', data_type.alias)

    def test_normalize_failure(self):
        with self.assertRaises(ValueError) as cm:
            DataType.normalize('Gnartz')
        self.assertEqual("unknown data type 'Gnartz'",
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            # noinspection PyTypeChecker
            DataType.normalize(42)
        self.assertEqual("cannot convert 42 into a data type",
                         f'{cm.exception}')

    def test_equality(self):
        self.assertIs(DATASET_TYPE, DATASET_TYPE)
        self.assertEqual(DATASET_TYPE, DATASET_TYPE)

        self.assertEqual(copy.deepcopy(DATASET_TYPE), DATASET_TYPE)
        self.assertNotEqual(MULTI_LEVEL_DATASET_TYPE, DATASET_TYPE)

        self.assertEqual(hash(copy.deepcopy(DATASET_TYPE)), hash(DATASET_TYPE))
        self.assertNotEqual(hash(MULTI_LEVEL_DATASET_TYPE), hash(DATASET_TYPE))

    def test_is_sub_type_of(self):
        self.assertTrue(ANY_TYPE.is_sub_type_of(ANY_TYPE))
        self.assertFalse(ANY_TYPE.is_sub_type_of(DATASET_TYPE))
        self.assertFalse(ANY_TYPE.is_sub_type_of(MULTI_LEVEL_DATASET_TYPE))

        self.assertTrue(DATASET_TYPE.is_sub_type_of(ANY_TYPE))
        self.assertTrue(DATASET_TYPE.is_sub_type_of(DATASET_TYPE))
        self.assertFalse(DATASET_TYPE.is_sub_type_of(MULTI_LEVEL_DATASET_TYPE))

        a_type = DataType.normalize(A)
        b_type = DataType.normalize(B)
        c_type = DataType.normalize(C)
        self.assertTrue(a_type.is_sub_type_of(a_type))
        self.assertFalse(a_type.is_sub_type_of(b_type))
        self.assertFalse(a_type.is_sub_type_of(c_type))
        self.assertTrue(b_type.is_sub_type_of(a_type))
        self.assertTrue(b_type.is_sub_type_of(b_type))
        self.assertFalse(b_type.is_sub_type_of(c_type))
        self.assertFalse(c_type.is_sub_type_of(a_type))
        self.assertFalse(c_type.is_sub_type_of(b_type))
        self.assertTrue(c_type.is_sub_type_of(c_type))

    def test_is_super_type_of(self):
        self.assertTrue(ANY_TYPE.is_super_type_of(ANY_TYPE))
        self.assertTrue(ANY_TYPE.is_super_type_of(DATASET_TYPE))
        self.assertTrue(ANY_TYPE.is_super_type_of(DATASET_TYPE))

        self.assertTrue(ANY_TYPE.is_super_type_of(DATASET_TYPE))
        self.assertTrue(DATASET_TYPE.is_super_type_of(DATASET_TYPE))
        self.assertFalse(MULTI_LEVEL_DATASET_TYPE.is_super_type_of(DATASET_TYPE))

        a_type = DataType.normalize(A)
        b_type = DataType.normalize(B)
        c_type = DataType.normalize(C)
        self.assertTrue(a_type.is_super_type_of(a_type))
        self.assertTrue(a_type.is_super_type_of(b_type))
        self.assertFalse(a_type.is_super_type_of(c_type))
        self.assertFalse(b_type.is_super_type_of(a_type))
        self.assertTrue(b_type.is_super_type_of(b_type))
        self.assertFalse(b_type.is_super_type_of(c_type))
        self.assertFalse(c_type.is_super_type_of(a_type))
        self.assertFalse(c_type.is_super_type_of(b_type))
        self.assertTrue(c_type.is_super_type_of(c_type))

    def test_schema(self):
        self.assertIsInstance(DATASET_TYPE.get_schema(), JsonStringSchema)
