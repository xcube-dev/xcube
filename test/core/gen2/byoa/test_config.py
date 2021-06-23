import os.path
import unittest

from xcube.core.gen2.byoa.config import CodeConfig
from xcube.core.gen2.byoa.fileset import FileSet

INLINE_CODE = (
    'def process_dataset(ds, text="Hello"):\n'
    '    return ds.assign_attrs(comment=f"xcube says {text}")\n'
)


class CodeConfigTest(unittest.TestCase):

    def test_from_code_string(self):
        code_config = CodeConfig.from_code(
            INLINE_CODE,
            module_name='user_code_1',
            parameters={'text': 'good bye'}
        )
        self.assertIsInstance(code_config, CodeConfig)
        self.assertEqual(INLINE_CODE, code_config.inline_code)
        self.assertEqual({'text': 'good bye'}, code_config.parameters)
        self.assertRegexpMatches(code_config.callable_ref, 'user_code_1:process_dataset')
        self.assertTrue(callable(code_config.get_callable()))

    def test_from_code_with_function_refs(self):
        code_config = CodeConfig.from_code(
            code=[
                'import xarray as xr\n',
                modify_dataset,
                transform_dataset
            ],
            module_name='user_code_2',
            parameters={'text': 'good bye'}
        )
        self.assertIsInstance(code_config, CodeConfig)
        self.assertEqual(
            (
                'import xarray as xr\n'
                '\n'
                '\n'
                'def modify_dataset(ds, text="Hello"):\n'
                '    return transform_dataset(ds, text)\n'
                '\n'
                '\n'
                'def transform_dataset(ds, text):\n'
                '    return ds.assign_attrs(comment=f"xcube says {text}")\n'
            ),
            code_config.inline_code)
        self.assertIsNone(code_config.file_set)
        self.assertEqual({'text': 'good bye'}, code_config.parameters)
        self.assertEqual('user_code_2:modify_dataset', code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_from_callable(self):
        code_config = CodeConfig.from_callable(modify_dataset, parameters=dict(text='Good bye!'))
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsNone(code_config.inline_code)
        self.assertIsNone(code_config.file_set, FileSet)
        self.assertIsNone(code_config.callable_ref)
        self.assertIs(modify_dataset, code_config.get_callable())

    def test_from_file_set_dir(self):
        dir_path = os.path.join(os.path.dirname(__file__), 'test_data', 'user_code')
        code_config = CodeConfig.from_file_set(dir_path, 'processor:process_dataset')
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsInstance(code_config.file_set, FileSet)
        self.assertIsNone(code_config.inline_code)
        self.assertEqual('processor:process_dataset', code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_from_file_set_zip(self):
        dir_path = os.path.join(os.path.dirname(__file__), 'test_data', 'user_code.zip')
        code_config = CodeConfig.from_file_set(dir_path, 'processor:process_dataset')
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsInstance(code_config.file_set, FileSet)
        self.assertIsNone(code_config.inline_code)
        self.assertEqual('processor:process_dataset', code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_for_local_from_callable(self):
        user_code_config = CodeConfig.from_callable(modify_dataset)

        local_code_config = user_code_config.for_local()
        self.assertIs(user_code_config, local_code_config)
        self.assertIs(modify_dataset, local_code_config.get_callable())

    def test_for_local_from_inline_code(self):
        user_code_config = CodeConfig.from_code(INLINE_CODE)

        local_code_config = user_code_config.for_local()
        self.assertIsInstance(local_code_config, CodeConfig)
        self.assertIsInstance(local_code_config.callable_ref, str)
        self.assertIsNone(local_code_config._callable)
        self.assertIsNone(local_code_config.inline_code)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertTrue(local_code_config.file_set.is_local_dir())
        self.assertRegexpMatches(os.path.basename(local_code_config.file_set.path),
                                 'xcube-gen-byoa-*.')
        self.assertRegexpMatches(local_code_config.callable_ref,
                                 'xcube_gen_byoa_*.:process_dataset')
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_from_file_set_dir(self):
        dir_path = os.path.join(os.path.dirname(__file__),
                                'test_data', 'user_code')
        user_code_config = CodeConfig.from_file_set(dir_path,
                                                    'processor:process_dataset')

        local_code_config = user_code_config.for_local()
        self.assertIs(user_code_config, local_code_config)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertIs(user_code_config.file_set, local_code_config.file_set)
        self.assertEqual(dir_path, local_code_config.file_set.path)
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_from_file_set_zip(self):
        zip_path = os.path.join(os.path.dirname(__file__),
                                'test_data', 'user_code.zip')
        user_code_config = CodeConfig.from_file_set(zip_path,
                                                    'processor:process_dataset')

        local_code_config = user_code_config.for_local()
        self.assertIsInstance(local_code_config, CodeConfig)
        self.assertIsInstance(local_code_config.callable_ref, str)
        self.assertIsNone(local_code_config._callable)
        self.assertIsNone(local_code_config.inline_code)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertTrue(local_code_config.file_set.is_local_dir())
        self.assertRegexpMatches(os.path.basename(local_code_config.file_set.path),
                                 'xcube-gen-byoa-*.')
        self.assertEqual(local_code_config.callable_ref,
                         'processor:process_dataset')
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_illegal_state(self):
        code_config = CodeConfig.from_code(INLINE_CODE)
        code_config.inline_code = None
        with self.assertRaises(RuntimeError) as e:
            code_config.for_local()
        self.assertEqual(('for_local() failed due to an invalid CodeConfig state',),
                         e.exception.args)

    def test_for_service_from_callable(self):
        user_code_config = CodeConfig.from_callable(transform_dataset)

        service_code_config = user_code_config.for_service()
        self.assertIsInstance(service_code_config, CodeConfig)
        self.assertIsInstance(service_code_config.callable_ref, str)
        self.assertIsNone(service_code_config._callable)
        self.assertIsNone(service_code_config.inline_code)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertTrue(service_code_config.file_set.is_local_zip())

    def test_for_service_from_file_set_zip(self):
        zip_path = os.path.join(os.path.dirname(__file__),
                                'test_data', 'user_code.zip')
        user_code_config = CodeConfig.from_file_set(zip_path,
                                                    'processor:process_dataset')

        service_code_config = user_code_config.for_service()
        self.assertIs(user_code_config, service_code_config)
        self.assertIs(user_code_config.file_set, service_code_config.file_set)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertEqual(zip_path, service_code_config.file_set.path)

    def test_for_service_from_file_set_dir(self):
        dir_path = os.path.join(os.path.dirname(__file__),
                                'test_data', 'user_code')
        user_code_config = CodeConfig.from_file_set(dir_path,
                                                    'processor:process_dataset')

        service_code_config = user_code_config.for_service()
        self.assertIsInstance(service_code_config, CodeConfig)
        self.assertIsInstance(service_code_config.callable_ref, str)
        self.assertIsNone(service_code_config._callable)
        self.assertIsNone(service_code_config.inline_code)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertTrue(service_code_config.file_set.is_local_zip())
        self.assertRegexpMatches(os.path.basename(service_code_config.file_set.path),
                                 'xcube-gen-byoa-*.')
        self.assertEqual(service_code_config.callable_ref,
                         'processor:process_dataset')
        self.assertTrue(callable(service_code_config.get_callable()))

    def test_for_service_illegal_state(self):
        code_config = CodeConfig.from_code(INLINE_CODE)
        code_config.inline_code = None
        with self.assertRaises(RuntimeError) as e:
            code_config.for_service()
        self.assertEqual(('for_service() failed due to an invalid CodeConfig state',),
                         e.exception.args)

    def test_to_dict(self):
        d = CodeConfig.from_callable(modify_dataset).to_dict()
        # Shall we raise instead?
        self.assertEqual({}, d)

        d = CodeConfig.from_code(INLINE_CODE, module_name='user_code').to_dict()
        self.assertEqual(
            {
                'callable_ref': 'user_code:process_dataset',
                'inline_code': INLINE_CODE,
            },
            d
        )

        d = CodeConfig.from_file_set(FileSet('github://dcs4cop:xcube@v0.8.2.dev0',
                                             includes='*.py'),
                                     callable_ref=('test.core.gen2.byoa.test_config:'
                                                   'modify_dataset')).to_dict()
        self.assertIsInstance(d.get('file_set'), dict)
        self.assertEqual(('test.core.gen2.byoa.test_config:'
                          'modify_dataset'),
                         d.get('callable_ref'))


def modify_dataset(ds, text="Hello"):
    return transform_dataset(ds, text)


def transform_dataset(ds, text):
    return ds.assign_attrs(comment=f"xcube says {text}")
