import unittest

from test.sampledata import create_highroc_dataset
from xcube.utils import select_variables


class SelectVariablesTest(unittest.TestCase):
    def test_select_variables_for_none(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = select_variables(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = select_variables(ds1, set())
        self.assertIs(ds2, ds1)

    def test_select_variables_for_all(self):
        ds1 = create_highroc_dataset()
        ds2 = select_variables(ds1, {'*'})
        self.assertIs(ds2, ds1)
        ds2 = select_variables(ds1, {'bibo'})
        self.assertEqual(0, len(ds2.data_vars))

    def test_select_variables_for_some(self):
        ds1 = create_highroc_dataset()
        self.assertEqual(36, len(ds1.data_vars))
        ds2 = select_variables(ds1, {'conc_chl', 'c2rcc_flags', 'rtoa_10'})
        self.assertEqual(3, len(ds2.data_vars))
        ds2 = select_variables(ds1, {'c*'})
        self.assertEqual(2, len(ds2.data_vars))
        ds2 = select_variables(ds1, {'rtoa_*', 'rrs_*'})
        self.assertEqual(32, len(ds2.data_vars))
        ds2 = select_variables(ds1, {'rtoa_?', 'rrs_?'})
        self.assertEqual(18, len(ds2.data_vars))
        ds2 = select_variables(ds1, {'rtoa_1?', 'rrs_1?'})
        self.assertEqual(12, len(ds2.data_vars))
