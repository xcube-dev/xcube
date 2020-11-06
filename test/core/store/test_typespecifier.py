import geopandas as gpd
import unittest
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.store.typespecifier import TYPE_SPECIFIER_CUBE
from xcube.core.store.typespecifier import TYPE_SPECIFIER_DATASET
from xcube.core.store.typespecifier import TYPE_SPECIFIER_GEODATAFRAME
from xcube.core.store.typespecifier import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store.typespecifier import get_type_specifier
from xcube.core.store.typespecifier import TypeSpecifier

from xcube.core.mldataset import BaseMultiLevelDataset


class TypeSpecifierTest(unittest.TestCase):

    def test_name(self):
        type_specifier = TypeSpecifier(name='dataset')
        self.assertEqual('dataset', type_specifier.name)
        self.assertEqual(set(), type_specifier.flags)

    def test_name_and_flags(self):
        type_specifier = TypeSpecifier(name='dataset', flags={'cube', 'multilevel'})
        self.assertEqual('dataset', type_specifier.name)
        self.assertEqual({'cube', 'multilevel'}, type_specifier.flags)

    def test_string(self):
        type_specifier = TypeSpecifier(name='dataset', flags={'cube', 'multilevel'})
        self.assertEqual(str(type_specifier), 'dataset[cube,multilevel]')

    def test_equality(self):
        type_specifier = TypeSpecifier(name='dataset', flags={'cube'})
        self.assertFalse(type_specifier == TypeSpecifier('geodataframe'))
        self.assertFalse(type_specifier == TypeSpecifier('geodataframe', flags={'cube'}))
        self.assertFalse(type_specifier == TypeSpecifier('dataset'))
        self.assertFalse(type_specifier == TypeSpecifier('dataset', flags={'multilevel'}))
        self.assertFalse(type_specifier == TypeSpecifier('dataset', flags={'multilevel', 'cube'}))
        self.assertTrue(type_specifier == TypeSpecifier('dataset', flags={'cube'}))

    def test_equality_flag_order_is_irrelevant(self):
        type_specifier_1 = TypeSpecifier(name='dataset', flags={'cube', 'multilevel'})
        type_specifier_2 = TypeSpecifier(name='dataset', flags={'multilevel', 'cube'})
        self.assertTrue(type_specifier_1  == type_specifier_2)

    def test_equality_any(self):
        type_specifier = TypeSpecifier(name='*', flags={'cube'})
        self.assertFalse(type_specifier == TypeSpecifier('geodataframe'))
        self.assertFalse(type_specifier == TypeSpecifier('geodataframe', flags={'cube'}))
        self.assertFalse(type_specifier == TypeSpecifier('dataset'))
        self.assertFalse(type_specifier == TypeSpecifier('dataset', flags={'multilevel'}))
        self.assertFalse(type_specifier == TypeSpecifier('dataset', flags={'multilevel', 'cube'}))
        self.assertFalse(type_specifier == TypeSpecifier('dataset', flags={'cube'}))
        self.assertFalse(type_specifier == TypeSpecifier('*'))
        self.assertTrue(type_specifier == TypeSpecifier('*', flags={'cube'}))

    def test_is_compatible(self):
        type_specifier = TypeSpecifier(name='dataset', flags={'cube'})
        self.assertFalse(type_specifier.is_compatible(TypeSpecifier('geodataframe')))
        self.assertFalse(type_specifier.is_compatible(TypeSpecifier('geodataframe', flags={'cube'})))
        self.assertFalse(type_specifier.is_compatible(TypeSpecifier('dataset')))
        self.assertFalse(type_specifier.is_compatible(TypeSpecifier('dataset', flags={'multilevel'})))
        self.assertTrue(type_specifier.is_compatible(TypeSpecifier('dataset', flags={'multilevel', 'cube'})))
        self.assertTrue(type_specifier.is_compatible(TypeSpecifier('dataset', flags={'cube'})))

    def test_is_compatible_any(self):
        type_specifier = TypeSpecifier(name='*', flags={'cube'})
        self.assertFalse(type_specifier.is_compatible(TypeSpecifier('geodataframe')))
        self.assertTrue(type_specifier.is_compatible(TypeSpecifier('geodataframe', flags={'cube'})))
        self.assertFalse(type_specifier.is_compatible(TypeSpecifier('dataset')))
        self.assertFalse(type_specifier.is_compatible(TypeSpecifier('dataset', flags={'multilevel'})))
        self.assertTrue(type_specifier.is_compatible(TypeSpecifier('dataset', flags={'cube'})))
        self.assertTrue(type_specifier.is_compatible(TypeSpecifier('dataset', flags={'multilevel', 'cube'})))

    def test_normalize(self):
        dataset_flagged_specifier = TypeSpecifier(name='dataset', flags={'cube'})
        self.assertEqual(TypeSpecifier.normalize(dataset_flagged_specifier), dataset_flagged_specifier)
        self.assertEqual(TypeSpecifier.normalize('dataset[cube]'), dataset_flagged_specifier)

        dataset_specifier = TypeSpecifier(name='dataset')
        self.assertEqual(TypeSpecifier.normalize(dataset_specifier), dataset_specifier)
        self.assertEqual(TypeSpecifier.normalize('dataset'), dataset_specifier)

    def test_parse(self):
        parsed_specifier = TypeSpecifier.parse('dataset[cube,multilevel]')
        self.assertEqual('dataset', parsed_specifier.name)
        self.assertEqual({'cube', 'multilevel'}, parsed_specifier.flags)

    def test_parse_exception(self):
        with self.assertRaises(SyntaxError) as cm:
            TypeSpecifier.parse('An unparseable expression[')
        self.assertEqual('"An unparseable expression[" cannot be parsed: No end brackets found', f'{cm.exception}')


class GetTypeSpecifierTest(unittest.TestCase):

    def test_get_type_specifier(self):
        self.assertIsNone(get_type_specifier(dict()))
        self.assertEqual(get_type_specifier(new_cube()), TYPE_SPECIFIER_CUBE)
        self.assertEqual(get_type_specifier(xr.Dataset()), TYPE_SPECIFIER_DATASET)
        self.assertEqual(get_type_specifier(BaseMultiLevelDataset(xr.Dataset())), TYPE_SPECIFIER_MULTILEVEL_DATASET)
        self.assertEqual(get_type_specifier(gpd.GeoDataFrame()), TYPE_SPECIFIER_GEODATAFRAME)
