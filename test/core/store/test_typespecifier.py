import unittest

import geopandas as gpd
import xarray as xr

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.new import new_cube
from xcube.core.store.typespecifier import TYPE_SPECIFIER_CUBE
from xcube.core.store.typespecifier import TYPE_SPECIFIER_DATASET
from xcube.core.store.typespecifier import TYPE_SPECIFIER_GEODATAFRAME
from xcube.core.store.typespecifier import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store.typespecifier import TypeSpecifier
from xcube.core.store.typespecifier import get_type_specifier


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
        self.assertTrue(type_specifier_1 == type_specifier_2)

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

    def test_it_satisfies_without_flags(self):
        type_specifier = TypeSpecifier(name='dataset')
        self._assert_it_not_satisfies(type_specifier, TypeSpecifier('dataset', flags={'cube'}))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('dataset'))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('*'))

    def test_it_satisfies_with_flags(self):
        type_specifier = TypeSpecifier(name='dataset', flags={'cube'})
        self._assert_it_not_satisfies(type_specifier, TypeSpecifier('geodataframe'))
        self._assert_it_not_satisfies(type_specifier, TypeSpecifier('geodataframe', flags={'cube'}))
        self._assert_it_not_satisfies(type_specifier, TypeSpecifier('dataset', flags={'multilevel'}))
        self._assert_it_not_satisfies(type_specifier, TypeSpecifier('dataset', flags={'multilevel', 'cube'}))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('dataset'))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('dataset', flags={'cube'}))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('*'))

    def test_it_satisfies_as_any(self):
        type_specifier = TypeSpecifier(name='*')
        self._assert_it_satisfies(type_specifier, TypeSpecifier('geodataframe'))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('geodataframe', flags={'cube'}))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('dataset'))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('dataset', flags={'cube'}))
        self._assert_it_satisfies(type_specifier, TypeSpecifier('*'))

    def _assert_it_satisfies(self, a: TypeSpecifier, b: TypeSpecifier):
        self.assertTrue(a.satisfies(b), f'"{a}" did unexpectedly not satisfy "{b}"')
        self.assertTrue(b.is_satisfied_by(a), f'"{b}" is unexpectedly not satisfied by "{a}"')

    def _assert_it_not_satisfies(self, a: TypeSpecifier, b: TypeSpecifier):
        self.assertFalse(a.satisfies(b), f'"{a}" did unexpectedly satisfy "{b}"')
        self.assertFalse(b.is_satisfied_by(a), f'"{b}" is unexpectedly satisfied by "{a}"')

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
