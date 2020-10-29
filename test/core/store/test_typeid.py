import geopandas as gpd
import unittest
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.store.typeid import TYPE_ID_CUBE
from xcube.core.store.typeid import TYPE_ID_DATASET
from xcube.core.store.typeid import TYPE_ID_GEO_DATA_FRAME
from xcube.core.store.typeid import TYPE_ID_MULTI_LEVEL_DATASET
from xcube.core.store.typeid import get_type_id
from xcube.core.store.typeid import TypeId

from xcube.core.mldataset import BaseMultiLevelDataset


class TypeIdTest(unittest.TestCase):

    def test_name(self):
        type_id = TypeId(name='dataset')
        self.assertEqual('dataset', type_id.name)
        self.assertEqual(set(), type_id.flags)

    def test_name_and_flags(self):
        type_id = TypeId(name='dataset', flags={'cube', 'multilevel'})
        self.assertEqual('dataset', type_id.name)
        self.assertEqual({'cube', 'multilevel'}, type_id.flags)

    def test_string(self):
        type_id = TypeId(name='dataset', flags={'cube', 'multilevel'})
        self.assertEqual(str(type_id), 'dataset[cube,multilevel]')

    def test_equality(self):
        type_id = TypeId(name='dataset', flags={'cube'})
        self.assertFalse(type_id == TypeId('geodataframe'))
        self.assertFalse(type_id == TypeId('geodataframe', flags={'cube'}))
        self.assertFalse(type_id == TypeId('dataset'))
        self.assertFalse(type_id == TypeId('dataset', flags={'multilevel'}))
        self.assertFalse(type_id == TypeId('dataset', flags={'multilevel', 'cube'}))
        self.assertTrue(type_id == TypeId('dataset', flags={'cube'}))

    def test_equality_flag_order_is_irrelevant(self):
        type_id_1 = TypeId(name='dataset', flags={'cube', 'multilevel'})
        type_id_2 = TypeId(name='dataset', flags={'multilevel', 'cube'})
        self.assertTrue(type_id_1  == type_id_2)

    def test_equality_any(self):
        type_id = TypeId(name='*', flags={'cube'})
        self.assertFalse(type_id == TypeId('geodataframe'))
        self.assertFalse(type_id == TypeId('geodataframe', flags={'cube'}))
        self.assertFalse(type_id == TypeId('dataset'))
        self.assertFalse(type_id == TypeId('dataset', flags={'multilevel'}))
        self.assertFalse(type_id == TypeId('dataset', flags={'multilevel', 'cube'}))
        self.assertFalse(type_id == TypeId('dataset', flags={'cube'}))
        self.assertFalse(type_id == TypeId('*'))
        self.assertTrue(type_id == TypeId('*', flags={'cube'}))

    def test_is_compatible(self):
        type_id = TypeId(name='dataset', flags={'cube'})
        self.assertFalse(type_id.is_compatible(TypeId('geodataframe')))
        self.assertFalse(type_id.is_compatible(TypeId('geodataframe', flags={'cube'})))
        self.assertFalse(type_id.is_compatible(TypeId('dataset')))
        self.assertFalse(type_id.is_compatible(TypeId('dataset', flags={'multilevel'})))
        self.assertTrue(type_id.is_compatible(TypeId('dataset', flags={'multilevel', 'cube'})))
        self.assertTrue(type_id.is_compatible(TypeId('dataset', flags={'cube'})))

    def test_is_compatible_any(self):
        type_id = TypeId(name='*', flags={'cube'})
        self.assertFalse(type_id.is_compatible(TypeId('geodataframe')))
        self.assertTrue(type_id.is_compatible(TypeId('geodataframe', flags={'cube'})))
        self.assertFalse(type_id.is_compatible(TypeId('dataset')))
        self.assertFalse(type_id.is_compatible(TypeId('dataset', flags={'multilevel'})))
        self.assertTrue(type_id.is_compatible(TypeId('dataset', flags={'cube'})))
        self.assertTrue(type_id.is_compatible(TypeId('dataset', flags={'multilevel', 'cube'})))

    def test_normalize(self):
        dataset_flagged_id = TypeId(name='dataset', flags={'cube'})
        self.assertEqual(TypeId.normalize(dataset_flagged_id), dataset_flagged_id)
        self.assertEqual(TypeId.normalize('dataset[cube]'), dataset_flagged_id)

        dataset_id = TypeId(name='dataset')
        self.assertEqual(TypeId.normalize(dataset_id), dataset_id)
        self.assertEqual(TypeId.normalize('dataset'), dataset_id)

    def test_parse(self):
        parsed_id = TypeId.parse('dataset[cube,multilevel]')
        self.assertEqual('dataset', parsed_id.name)
        self.assertEqual({'cube', 'multilevel'}, parsed_id.flags)

    def test_parse_exception(self):
        with self.assertRaises(SyntaxError) as cm:
            TypeId.parse('An unparseable expression[')
        self.assertEqual('"An unparseable expression[" cannot be parsed: No end brackets found', f'{cm.exception}')


class GetTypeIdTest(unittest.TestCase):

    def test_get_type_id(self):
        self.assertIsNone(get_type_id(dict()))
        self.assertEqual(get_type_id(new_cube()), TYPE_ID_CUBE)
        self.assertEqual(get_type_id(xr.Dataset()), TYPE_ID_DATASET)
        self.assertEqual(get_type_id(BaseMultiLevelDataset(xr.Dataset())), TYPE_ID_MULTI_LEVEL_DATASET)
        self.assertEqual(get_type_id(gpd.GeoDataFrame()), TYPE_ID_GEO_DATA_FRAME)
