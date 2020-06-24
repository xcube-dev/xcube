import unittest

from xcube.core.store.descriptor import VariableDescriptor


class VariableDescriptorTest(unittest.TestCase):

    def test_variable_descriptor_basic(self):
        vd1 = VariableDescriptor('gz', 'zughysz', ['rtdswgt', 'dref', 'zdrs5ge'])
        self.assertEqual('gz', vd1.name)
        self.assertEqual('zughysz', vd1.dtype)
        self.assertEqual(('rtdswgt', 'dref', 'zdrs5ge'), vd1.dims)
        self.assertEqual(3, vd1.ndim)
        self.assertEqual(None, vd1.attrs)

        vd3 = VariableDescriptor('gz', 'zughysz', ['rtdswgt', 'dref', 'zdrs5ge'], {'d': 2, 'zjgu': ''})
        self.assertEqual('gz', vd3.name)
        self.assertEqual('zughysz', vd3.dtype)
        self.assertEqual(('rtdswgt', 'dref', 'zdrs5ge'), vd3.dims)
        self.assertEqual(3, vd3.ndim)
        self.assertEqual({'d': 2, 'zjgu': ''}, vd3.attrs)

    def test_variable_descriptor_to_dict(self):
        vd = VariableDescriptor('xf', 'rj', ['dfjhrt', 'sg'], {'ssd': 4, 'zjgrhgu': 'dgfrf'})
        expected = {
            'name': 'xf',
            'dtype': 'rj',
            'dims': ('dfjhrt', 'sg'),
            'ndim': 2,
            'attrs': {
                'ssd': 4,
                'zjgrhgu': 'dgfrf'
            }
        }
        self.assertEqual(expected, vd.to_dict())

    def test_variable_descriptor_from_dict(self):
        vd_as_dict = {
            'name': 'xf',
            'dtype': 'rj',
            'dims': ('dfjhrt', 'sg'),
            'ndim': 2,
            'attrs': {
                'ssd': 4,
                'zjgrhgu': 'dgfrf'
            }
        }
        vd = VariableDescriptor.from_dict(vd_as_dict)
        self.assertEqual('xf', vd.name)
        self.assertEqual('rj', vd.dtype)
        self.assertEqual(('dfjhrt', 'sg'), vd.dims)
        self.assertEqual(2, vd.ndim)
        self.assertEqual({'ssd': 4, 'zjgrhgu': 'dgfrf'}, vd.attrs)

        vd_fail = None
        try:
            vd_fail = VariableDescriptor.from_dict({'name': 'dhgfr', 'dtype': 'ghdst'})
            self.fail('Should not come here')
        except ValueError:
            self.assertIsNone(vd_fail)

        try:
            vd_fail = VariableDescriptor.from_dict({'name': 'dhgfr', 'dims': ['faer', 'bjunda']})
            self.fail('Should not come here')
        except ValueError:
            self.assertIsNone(vd_fail)

        try:
            vd_fail = VariableDescriptor.from_dict({'dtype': 'ghdst', 'dims': ['faer', 'bjunda']})
            self.fail('Should not come here')
        except ValueError:
            self.assertIsNone(vd_fail)
