import unittest

from xcube.core.store.descriptor import DatasetDescriptor
from xcube.core.store.descriptor import VariableDescriptor


class DatasetDescriptorTest(unittest.TestCase):

    def test_from_dict_no_data_id(self):
        try:
            descriptor_dict = dict()
            DatasetDescriptor.from_dict(descriptor_dict)
            self.fail('Exception expected')
        except ValueError:
            pass

    def test_from_dict_wrong_type_id(self):
        try:
            descriptor_dict = dict(data_id='xyz', type_id='tsr')
            DatasetDescriptor.from_dict(descriptor_dict)
            self.fail('Exception expected')
        except ValueError:
            pass

    def test_from_dict_basic(self):
        descriptor_dict = dict(data_id='xyz')
        descriptor = DatasetDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('dataset', descriptor.type_id)

    def test_from_dict_derived_type(self):
        descriptor_dict = dict(data_id='xyz', type_id='dataset[fegd]')
        descriptor = DatasetDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('dataset[fegd]', descriptor.type_id)

    def test_from_dict_full(self):
        descriptor_dict = dict(data_id='xyz',
                               crs='EPSG:9346',
                               bbox=(10., 20., 30., 40.),
                               spatial_res=20.,
                               time_range=('2017-06-05', '2017-06-27'),
                               time_period='daily',
                               dims=dict(x=1, y=2, z=3),
                               data_vars=[dict(name='xf',
                                               dtype='rj',
                                               dims=('dfjhrt', 'sg'),
                                               ndim=2,
                                               attrs=dict(ssd=4,
                                                          zjgrhgu='dgfrf'
                                                          )
                                               )
                                          ],
                               attrs=dict(dzus=236,
                                          tgr7h='rt5',
                                          df='s8fd4w5'
                                          ),
                               open_params_schema=dict(type="object",
                                                       properties=dict(variable_names=dict(type='array',
                                                                                           items=dict(type='string')
                                                                                           )
                                                                       )
                                                       )
                               )
        descriptor = DatasetDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('dataset', descriptor.type_id)
        self.assertEqual('EPSG:9346', descriptor.crs)
        self.assertEqual((10., 20., 30., 40.), descriptor.bbox)
        self.assertEqual(20., descriptor.spatial_res)
        self.assertEqual(('2017-06-05', '2017-06-27'), descriptor.time_range)
        self.assertEqual('daily', descriptor.time_period)
        self.assertEqual(dict(x=1, y=2, z=3), descriptor.dims)
        self.assertEqual(1, len(descriptor.data_vars))
        self.assertEqual(236, descriptor.attrs.get('dzus', None))
        self.assertEqual('rt5', descriptor.attrs.get('tgr7h', None))
        self.assertEqual('s8fd4w5', descriptor.attrs.get('df', None))
        self.assertEqual('object', descriptor.open_params_schema.get('type', None))


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
