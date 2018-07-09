import unittest

from test.sampledata import create_highroc_dataset
from xcube.utils import select_variables, to_key_dict_pair, update_variable_props


class SelectVariablesTest(unittest.TestCase):
    def test_select_all(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = select_variables(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = select_variables(ds1, ['*'])
        self.assertIs(ds2, ds1)

    def test_select_none(self):
        ds1 = create_highroc_dataset()
        ds2 = select_variables(ds1, [])
        self.assertEqual(0, len(ds2.data_vars))
        ds2 = select_variables(ds1, ['bibo'])
        self.assertEqual(0, len(ds2.data_vars))

    def test_select_variables_for_some(self):
        ds1 = create_highroc_dataset()
        self.assertEqual(36, len(ds1.data_vars))
        ds2 = select_variables(ds1, ['conc_chl', 'c2rcc_flags', 'rtoa_10'])
        self.assertEqual(3, len(ds2.data_vars))
        ds2 = select_variables(ds1, ['c*'])
        self.assertEqual(2, len(ds2.data_vars))
        ds2 = select_variables(ds1, ['rtoa_*', 'rrs_*'])
        self.assertEqual(32, len(ds2.data_vars))
        ds2 = select_variables(ds1, ['rtoa_?', 'rrs_?'])
        self.assertEqual(18, len(ds2.data_vars))
        ds2 = select_variables(ds1, ['rtoa_1?', 'rrs_1?'])
        self.assertEqual(12, len(ds2.data_vars))

    def test_select_variables(self):
        ds1 = create_highroc_dataset()
        self.assertEqual(36, len(ds1.data_vars))

        ds2 = select_variables(ds1, [{'conc_chl': 'chl_c2rcc'},
                                     {'c2rcc_flags': {'name': 'flags'}},
                                     'rtoa_10'])
        self.assertEqual(3, len(ds2.data_vars))
        self.assertIn('conc_chl', ds2.data_vars)
        self.assertIn('c2rcc_flags', ds2.data_vars)
        self.assertIn('rtoa_10', ds2.data_vars)


class UpdateVariablePropsTest(unittest.TestCase):
    def test_no_change(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = update_variable_props(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = update_variable_props(ds1, [])
        self.assertIs(ds2, ds1)

    def test_change_all_or_none(self):
        ds1 = create_highroc_dataset()
        ds2 = update_variable_props(ds1, [{'*': {'marker': True}}])
        self.assertEqual(len(ds1.data_vars), len(ds2.data_vars))
        self.assertTrue(all(['marker' in ds2[n].attrs for n in ds2.variables]))
        ds2 = update_variable_props(ds1, [{'bibo': {'marker': True}}])
        self.assertFalse(any(['marker' in ds2[n].attrs for n in ds2.variables]))

    def test_change_some(self):
        ds1 = create_highroc_dataset()
        ds2 = update_variable_props(ds1, [{'conc_chl': 'chl_c2rcc'},
                                          {'c2rcc_flags': {'name': 'flags',
                                                           'marker': True}},
                                          'rtoa_10'])

        self.assertEqual(len(ds1.data_vars), len(ds2.data_vars))

        self.assertNotIn('conc_chl', ds2.data_vars)
        self.assertNotIn('c2rcc_flags', ds2.data_vars)

        self.assertIn('chl_c2rcc', ds2.data_vars)
        self.assertIn('original_name', ds2.chl_c2rcc.attrs)
        self.assertEqual('conc_chl', ds2.chl_c2rcc.attrs['original_name'])

        self.assertIn('flags', ds2.data_vars)
        self.assertIn('original_name', ds2.flags.attrs)
        self.assertEqual('c2rcc_flags', ds2.flags.attrs['original_name'])
        self.assertIn('marker', ds2.flags.attrs)
        self.assertEqual(True, ds2.flags.attrs['marker'])

        self.assertIn('rtoa_10', ds2.data_vars)

        with self.assertRaises(ValueError) as cm:
            update_variable_props(ds1, ['conc_chl',
                                        'c2rcc_flags',
                                        {'rtoa_*': 'refl_toa_*'}])
        self.assertEqual("variable pattern 'rtoa_*' cannot be renamed into 'refl_toa_*'", f'{cm.exception}')


class ToKeyDictPairTest(unittest.TestCase):
    def test_just_key(self):
        t = to_key_dict_pair('a', default_key='name')
        self.assertEqual(('a', None), t)
        t = to_key_dict_pair('a?', default_key='name')
        self.assertEqual(('a?', None), t)

    def test_tuple(self):
        t = to_key_dict_pair(('a', 'udu'), default_key='name')
        self.assertEqual(('a', dict(name='udu')), t)
        t = to_key_dict_pair(('a*', 'udu'), default_key='name')
        self.assertEqual(('a*', dict(name='udu')), t)

    def test_dict(self):
        t = to_key_dict_pair('a', parent=dict(a='udu'), default_key='name')
        self.assertEqual(('a', dict(name='udu')), t)
        t = to_key_dict_pair('a*', parent={'a*': 'udu'}, default_key='name')
        self.assertEqual(('a*', dict(name='udu')), t)

    def test_mapping(self):
        t = to_key_dict_pair({'a': 'udu'}, default_key='name')
        self.assertEqual(('a', dict(name='udu')), t)
        t = to_key_dict_pair({'a*': dict(name='udu')})
        self.assertEqual(('a*', dict(name='udu')), t)
