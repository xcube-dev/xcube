import unittest

import numpy as np

from xcube.core.new import new_cube
from xcube.core.store.descriptor import DataDescriptor
from xcube.core.store.descriptor import DatasetDescriptor
from xcube.core.store.descriptor import GeoDataFrameDescriptor
from xcube.core.store.descriptor import VariableDescriptor
from xcube.core.store.descriptor import new_data_descriptor
from xcube.core.store.typespecifier import TypeSpecifier
from xcube.util.jsonschema import JsonObjectSchema


class NewDataDescriptorTest(unittest.TestCase):

    def test_new_dataset_descriptor(self):
        cube = new_cube(variables=dict(a=4.1, b=7.4))
        descriptor = new_data_descriptor('cube', cube)
        self.assertIsNotNone(descriptor)
        self.assertTrue(isinstance(descriptor, DatasetDescriptor))
        self.assertEqual('cube', descriptor.data_id)
        self.assertEqual('dataset[cube]', descriptor.type_specifier)
        self.assertEqual((-180.0, -90.0, 180.0, 90.0), descriptor.bbox)
        self.assertIsNone(descriptor.open_params_schema)
        self.assertEqual(('2010-01-01T00:00:00', '2010-01-06T00:00:00'), descriptor.time_range)
        self.assertEqual('1D', descriptor.time_period)
        self.assertEqual(1.0, descriptor.spatial_res)
        self.assertIsNotNone(descriptor.coords)
        self.assertEqual({'time': 5, 'lat': 180, 'lon': 360, 'bnds': 2}, descriptor.dims)
        self.assertIsNotNone(descriptor.data_vars)


class DataDescriptorTest(unittest.TestCase):

    def test_from_dict_no_data_id(self):
        with self.assertRaises(ValueError):
            descriptor_dict = dict()
            DataDescriptor.from_dict(descriptor_dict)

    def test_from_dict_no_type_specifier(self):
        with self.assertRaises(ValueError):
            descriptor_dict = dict(data_id='id')
            DataDescriptor.from_dict(descriptor_dict)

    def test_from_dict_random_type_specifier(self):
        descriptor_dict = dict(data_id='xyz', type_specifier='tsr')
        descriptor = DataDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('tsr', descriptor.type_specifier)

    def test_from_dict_dataset_type_specifier(self):
        descriptor_dict = dict(data_id='xyz', type_specifier='dataset')
        descriptor = DataDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertTrue(DatasetDescriptor, type(descriptor))
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('dataset', descriptor.type_specifier)

    def test_from_dict_geodataframe_type_specifier(self):
        descriptor_dict = dict(data_id='xyz', type_specifier='geodataframe')
        descriptor = DataDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertTrue(GeoDataFrameDescriptor, type(descriptor))
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('geodataframe', descriptor.type_specifier)

    def test_from_dict_full(self):
        descriptor_dict = dict(
            data_id='xyz',
            type_specifier='tsr',
            crs='EPSG:9346',
            bbox=(10., 20., 30., 40.),
            time_range=('2017-06-05', '2017-06-27'),
            time_period='daily',
            open_params_schema=dict(
                type="object",
                properties=dict(
                    variable_names=dict(
                        type='array',
                        items=dict(
                            type='string')
                    )
                )
            )
        )
        descriptor = DataDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('tsr', descriptor.type_specifier)
        self.assertEqual('EPSG:9346', descriptor.crs)
        self.assertEqual((10., 20., 30., 40.), descriptor.bbox)
        self.assertEqual(('2017-06-05', '2017-06-27'), descriptor.time_range)
        self.assertEqual('daily', descriptor.time_period)
        self.assertEqual('object', descriptor.open_params_schema.get('type', None))

    def test_to_dict(self):
        descriptor = DatasetDescriptor(data_id='xyz',
                                       type_specifier=TypeSpecifier('dataset', flags={'cube'}),
                                       crs='EPSG:9346',
                                       bbox=(10., 20., 30., 40.),
                                       spatial_res=20.,
                                       time_range=('2017-06-05', '2017-06-27'),
                                       time_period='daily')
        descriptor_dict = descriptor.to_dict()
        self.assertEqual(dict(data_id='xyz',
                              type_specifier='dataset[cube]',
                              crs='EPSG:9346',
                              bbox=(10., 20., 30., 40.),
                              spatial_res=20.,
                              time_range=('2017-06-05', '2017-06-27'),
                              time_period='daily'),
                         descriptor_dict)


class DatasetDescriptorTest(unittest.TestCase):

    def test_get_schema(self):
        schema = DatasetDescriptor.get_schema()
        self.assertIsInstance(schema, JsonObjectSchema)

    def test_from_dict_no_data_id(self):
        try:
            descriptor_dict = dict()
            DatasetDescriptor.from_dict(descriptor_dict)
            self.fail('Exception expected')
        except ValueError:
            pass

    def test_from_dict_wrong_type_specifier(self):
        with self.assertRaises(ValueError) as cm:
            descriptor_dict = dict(data_id='xyz', type_specifier='tsr')
            DatasetDescriptor.from_dict(descriptor_dict)
        self.assertEqual('type_specifier must satisfy type specifier "dataset", but was "tsr"',
                         f'{cm.exception}')

    def test_from_dict_basic(self):
        descriptor_dict = dict(data_id='xyz')
        descriptor = DatasetDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('dataset', descriptor.type_specifier)

    def test_from_dict_derived_type(self):
        descriptor_dict = dict(data_id='xyz', type_specifier='dataset[fegd]')
        descriptor = DatasetDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('dataset[fegd]', descriptor.type_specifier)

    def test_from_dict_full(self):
        descriptor_dict = dict(
            data_id='xyz',
            crs='EPSG:9346',
            bbox=(10., 20., 30., 40.),
            spatial_res=20.,
            time_range=('2017-06-05', '2017-06-27'),
            time_period='daily',
            coords=dict(
                rtdt=dict(
                    name='rtdt',
                    dtype='rj',
                    dims=('rtdt'),
                    ndim=2,
                    attrs=dict(
                        ssd=6,
                        zjgrhgu='hgtr'
                    )
                )
            ),
            dims=dict(x=1, y=2, z=3),
            data_vars=dict(
                xf=dict(
                    name='xf',
                    dtype='rj',
                    dims=('dfjhrt', 'sg'),
                    chunks=(2, 3),
                    ndim=2,
                    attrs=dict(
                        ssd=4,
                        zjgrhgu='dgfrf'
                    )
                )
            ),
            attrs=dict(
                dzus=236,
                tgr7h='rt5',
                df='s8fd4w5'
            ),
            open_params_schema=dict(
                type="object",
                properties=dict(
                    variable_names=dict(
                        type='array',
                        items=dict(
                            type='string'
                        )
                    )
                )
            )
        )
        descriptor = DatasetDescriptor.from_dict(descriptor_dict)
        self.assertIsNotNone(descriptor)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('dataset', descriptor.type_specifier)
        self.assertEqual('EPSG:9346', descriptor.crs)
        self.assertEqual((10., 20., 30., 40.), descriptor.bbox)
        self.assertEqual(20., descriptor.spatial_res)
        self.assertEqual(('2017-06-05', '2017-06-27'), descriptor.time_range)
        self.assertEqual('daily', descriptor.time_period)
        self.assertEqual(1, len(descriptor.coords))
        self.assertTrue('rtdt' in descriptor.coords)
        self.assertEqual(dict(x=1, y=2, z=3), descriptor.dims)
        self.assertEqual(1, len(descriptor.data_vars))
        self.assertTrue('xf' in descriptor.data_vars)
        self.assertEqual(236, descriptor.attrs.get('dzus', None))
        self.assertEqual('rt5', descriptor.attrs.get('tgr7h', None))
        self.assertEqual('s8fd4w5', descriptor.attrs.get('df', None))
        self.assertEqual('object', descriptor.open_params_schema.get('type', None))

    def test_from_dict_var_descriptors_as_dict(self):
        descriptor_dict = dict(
            data_id='xyz',
            crs='EPSG:9346',
            data_vars=dict(
                xf=dict(
                    name='xf',
                    dtype='rj',
                    dims=('dfjhrt', 'sg'),
                    chunks=(2, 3),
                    ndim=2,
                    attrs=dict(
                        ssd=4,
                        zjgrhgu='dgfrf'
                    )
                )
            )
        )
        descriptor = DatasetDescriptor.from_dict(descriptor_dict)
        self.assertEqual('xyz', descriptor.data_id)
        self.assertEqual('dataset', descriptor.type_specifier)
        self.assertEqual('EPSG:9346', descriptor.crs)
        self.assertEqual(1, len(descriptor.data_vars))
        self.assertTrue('xf' in descriptor.data_vars)
        self.assertTrue(type(descriptor.data_vars.get('xf')) == VariableDescriptor)

    def test_to_dict(self):
        coords = dict(
            rtdt=VariableDescriptor(
                name='rtdt',
                dtype='rj',
                dims=('rtdt',),
                chunks=(2,),
                attrs=dict(
                    ssd=6,
                    zjgrhgu='hgtr'
                )
            )
        )
        var_descriptors = dict(
            xf=VariableDescriptor(
                name='xf',
                dtype='rj',
                dims=('dfjhrt', 'sg'),
                chunks=(2, 3),
                attrs=dict(
                    ssd=4,
                    zjgrhgu='dgfrf'
                )
            )
        )
        descriptor = DatasetDescriptor(
            data_id='xyz',
            type_specifier=TypeSpecifier('dataset', flags={'cube'}),
            crs='EPSG:9346',
            bbox=(10., 20., 30., 40.),
            spatial_res=20.,
            time_range=('2017-06-05', '2017-06-27'),
            time_period='daily',
            coords=coords,
            dims=dict(
                x=1, y=2, z=3
            ),
            data_vars=var_descriptors,
            attrs=dict(
                dzus=236,
                tgr7h='rt5',
                df='s8fd4w5'
            )
        )
        descriptor_dict = descriptor.to_dict()
        self.assertEqual(
            dict(
                data_id='xyz',
                type_specifier='dataset[cube]',
                crs='EPSG:9346',
                bbox=(10., 20., 30., 40.),
                spatial_res=20.,
                time_range=('2017-06-05', '2017-06-27'),
                time_period='daily',
                coords=dict(
                    rtdt=dict(
                        name='rtdt',
                        dtype='rj',
                        dims=('rtdt',),
                        chunks=(2,),
                        ndim=1,
                        attrs=dict(
                            ssd=6,
                            zjgrhgu='hgtr'
                        )
                    )
                ),
                dims=dict(x=1, y=2, z=3),
                data_vars=dict(
                    xf=dict(
                        name='xf',
                        dtype='rj',
                        dims=('dfjhrt', 'sg'),
                        chunks=(2, 3),
                        ndim=2,
                        attrs=dict(
                            ssd=4,
                            zjgrhgu='dgfrf'
                        )
                    )
                ),
                attrs=dict(
                    dzus=236,
                    tgr7h='rt5',
                    df='s8fd4w5'
                )
            ),
            descriptor_dict
        )


class VariableDescriptorTest(unittest.TestCase):

    def test_variable_descriptor_basic(self):
        vd1 = VariableDescriptor('gz',
                                 'zughysz',
                                 ['rtdswgt', 'dref', 'zdrs5ge'],
                                 (3, 321, 4))
        self.assertEqual('gz', vd1.name)
        self.assertEqual('zughysz', vd1.dtype)
        self.assertEqual(('rtdswgt', 'dref', 'zdrs5ge'), vd1.dims)
        self.assertEqual((3, 321, 4), vd1.chunks)
        self.assertEqual(3, vd1.ndim)
        self.assertEqual(None, vd1.attrs)

        vd3 = VariableDescriptor('gz',
                                 'zughysz',
                                 ['rtdswgt', 'dref', 'zdrs5ge'],
                                 (3, 321, 4),
                                 {'d': 2, 'zjgu': ''})
        self.assertEqual('gz', vd3.name)
        self.assertEqual('zughysz', vd3.dtype)
        self.assertEqual(('rtdswgt', 'dref', 'zdrs5ge'), vd3.dims)
        self.assertEqual((3, 321, 4), vd3.chunks)
        self.assertEqual(3, vd3.ndim)
        self.assertEqual({'d': 2, 'zjgu': ''}, vd3.attrs)

    def test_variable_descriptor_to_dict(self):
        vd = VariableDescriptor('xf',
                                'rj',
                                ['dfjhrt', 'sg'],
                                (3, 2),
                                {'ssd': 4, 'zjgrhgu': 'dgfrf', 'fill_value': np.NaN})
        expected = {
            'name': 'xf',
            'dtype': 'rj',
            'dims': ('dfjhrt', 'sg'),
            'ndim': 2,
            'chunks': (3, 2),
            'attrs': {
                'ssd': 4,
                'zjgrhgu': 'dgfrf',
                'fill_value': None
            }
        }
        self.assertEqual(expected, vd.to_dict())

    def test_variable_descriptor_from_dict(self):
        vd_as_dict = {
            'name': 'xf',
            'dtype': 'rj',
            'dims': ('dfjhrt', 'sg'),
            'ndim': 2,
            'chunks':(3, 2),
            'attrs': {
                'ssd': 4,
                'zjgrhgu': 'dgfrf',
                'fill_value': None
            }
        }
        vd = VariableDescriptor.from_dict(vd_as_dict)
        self.assertEqual('xf', vd.name)
        self.assertEqual('rj', vd.dtype)
        self.assertEqual(('dfjhrt', 'sg'), vd.dims)
        self.assertEqual(2, vd.ndim)
        self.assertEqual((3, 2), vd.chunks)
        self.assertEqual({'ssd': 4, 'zjgrhgu': 'dgfrf', 'fill_value': None}, vd.attrs)

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
