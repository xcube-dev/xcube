import unittest

from test.sampledata import create_highroc_dataset
from xcube.api.select import select_vars


class SelectVariablesTest(unittest.TestCase):
    def test_select_all(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = select_vars(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = select_vars(ds1, ds1.data_vars.keys())
        self.assertIs(ds2, ds1)

    def test_select_none(self):
        ds1 = create_highroc_dataset()
        ds2 = select_vars(ds1, [])
        self.assertEqual(0, len(ds2.data_vars))
        ds2 = select_vars(ds1, ['bibo'])
        self.assertEqual(0, len(ds2.data_vars))

    def test_select_variables_for_some(self):
        ds1 = create_highroc_dataset()
        self.assertEqual(36, len(ds1.data_vars))
        ds2 = select_vars(ds1, ['conc_chl', 'c2rcc_flags', 'rtoa_10'])
        self.assertEqual(3, len(ds2.data_vars))
