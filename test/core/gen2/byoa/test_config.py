import os.path
import unittest

from xcube.core.gen2.byoa.config import CodeConfig
from xcube.core.gen2.byoa.fileset import FileSet

INLINE_CODE = (
    'def process_dataset(ds, text="Hello"):\n'
    '    return ds.assign_attrs(comment=f"xcube says {text}")\n'
)

PARENT_DIR = os.path.dirname(__file__)
LOCAL_MODULE_DIR = os.path.join(PARENT_DIR,
                                'test_data', 'user_code')
LOCAL_MODULE_ZIP = os.path.join(PARENT_DIR,
                                'test_data', 'user_code.zip')


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
        self.assertRegex(code_config.callable_ref, 'user_code_1:process_dataset')
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
        code_config = CodeConfig.from_callable(modify_dataset,
                                               parameters=dict(text='Good bye!'))
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsNone(code_config.inline_code)
        self.assertIsNone(code_config.file_set, FileSet)
        self.assertIsNone(code_config.callable_ref)
        self.assertIs(modify_dataset, code_config.get_callable())

    def test_from_file_set_dir(self):
        code_config = CodeConfig.from_file_set(LOCAL_MODULE_DIR,
                                               'processor:process_dataset')
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsInstance(code_config.file_set, FileSet)
        self.assertIsNone(code_config.inline_code)
        self.assertEqual('processor:process_dataset', code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_from_file_set_zip(self):
        code_config = CodeConfig.from_file_set(LOCAL_MODULE_ZIP,
                                               'processor:process_dataset')
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsInstance(code_config.file_set, FileSet)
        self.assertIsNone(code_config.inline_code)
        self.assertEqual('processor:process_dataset', code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_for_local_from_callable(self):
        user_code_config = CodeConfig.from_callable(modify_dataset)

        local_code_config = user_code_config.for_local()
        self.assertIs(user_code_config, local_code_config)
        # CodeConfigs from for_local() shall be able to load callable
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
        self.assertRegex(os.path.basename(local_code_config.file_set.path),
                         'xcube-gen-byoa-*.')
        self.assertRegex(local_code_config.callable_ref,
                         'xcube_gen_byoa_*.:process_dataset')
        # CodeConfigs from for_local() shall be able to load callable
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_from_file_set_dir(self):
        user_code_config = CodeConfig.from_file_set(LOCAL_MODULE_DIR,
                                                    'processor:process_dataset')

        local_code_config = user_code_config.for_local()
        self.assertIs(user_code_config, local_code_config)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertIs(user_code_config.file_set, local_code_config.file_set)
        self.assertEqual(LOCAL_MODULE_DIR, local_code_config.file_set.path)
        # CodeConfigs from for_local() shall be able to load callable
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_from_file_set_zip(self):
        user_code_config = CodeConfig.from_file_set(LOCAL_MODULE_ZIP,
                                                    'processor:process_dataset')

        local_code_config = user_code_config.for_local()
        self.assertIsInstance(local_code_config, CodeConfig)
        self.assertIsInstance(local_code_config.callable_ref, str)
        self.assertIsNone(local_code_config._callable)
        self.assertIsNone(local_code_config.inline_code)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertTrue(local_code_config.file_set.is_local_dir())
        self.assertRegex(os.path.basename(local_code_config.file_set.path),
                         'xcube-gen-byoa-*.')
        self.assertEqual(local_code_config.callable_ref,
                         'processor:process_dataset')
        # CodeConfigs from for_local() shall be able to load callable
        self.assertTrue(callable(local_code_config.get_callable()))

    @unittest.skipUnless(os.environ.get('ENABLE_XCUBE_BYOA_FILE_SET_REMOTE_TESTS') == '1',
                         'This is a manual test, '
                         'because it accesses the internet and it is slow.\n'
                         'To activate, set ENABLE_XCUBE_BYOA_FILE_SET_REMOTE_TESTS=1')
    def test_for_local_from_file_set_remote(self):
        # url = FileSet('zip::simplecache::https://github.com/dcs4cop/xcube/archive/v0.8.1.zip',
        #               parameters=dict(simplecache={'cache_storage': '.'}))
        url = FileSet('zip::simplecache::https://github.com/dcs4cop/xcube/archive/v0.8.1.zip',
                      parameters=dict(simplecache={'cache_storage': '.'}))
        # url = FileSet('github://dcs4cop:xcube@17a3a8526e0105ee610c86fcf5fe82fd62f9b273/',
        #               parameters=dict(username='forman',
        #                               token='...'))
        callable_ref = 'test.core.gen2.byoa.test_data.' \
                       'user_code.processor:process_dataset'
        user_code_config = CodeConfig.from_file_set(url,
                                                    callable_ref=callable_ref)

        local_code_config = user_code_config.for_local()
        self.assertIsInstance(local_code_config, CodeConfig)
        self.assertIsInstance(local_code_config.callable_ref, str)
        self.assertIsNone(local_code_config._callable)
        self.assertIsNone(local_code_config.inline_code)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertTrue(local_code_config.file_set.is_local_dir())
        self.assertEqual(url, local_code_config.file_set.path)
        self.assertEqual(callable_ref, local_code_config.callable_ref)
        # CodeConfigs from for_local() shall be able to load callable
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
        user_code_config = CodeConfig.from_file_set(LOCAL_MODULE_ZIP,
                                                    'processor:process_dataset')

        service_code_config = user_code_config.for_service()
        self.assertIs(user_code_config, service_code_config)
        self.assertIs(user_code_config.file_set, service_code_config.file_set)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertEqual(LOCAL_MODULE_ZIP, service_code_config.file_set.path)

    def test_for_service_from_file_set_dir(self):
        user_code_config = CodeConfig.from_file_set(LOCAL_MODULE_DIR,
                                                    'processor:process_dataset')

        service_code_config = user_code_config.for_service()
        self.assertIsInstance(service_code_config, CodeConfig)
        self.assertIsInstance(service_code_config.callable_ref, str)
        self.assertIsNone(service_code_config._callable)
        self.assertIsNone(service_code_config.inline_code)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertTrue(service_code_config.file_set.is_local_zip())
        self.assertRegex(os.path.basename(service_code_config.file_set.path),
                         'xcube-gen-byoa-*.')
        self.assertEqual(service_code_config.callable_ref,
                         'processor:process_dataset')

    def test_for_service_from_file_set_remote(self):
        url = 'https://github.com/dcs4cop/xcube/archive/refs/tags/v0.8.2.dev0.zip'
        callable_ref = 'test.core.gen2.byoa.test_data.' \
                       'user_code.processor:process_dataset'
        user_code_config = CodeConfig.from_file_set(url,
                                                    callable_ref=callable_ref)

        service_code_config = user_code_config.for_service()
        self.assertIsInstance(service_code_config, CodeConfig)
        self.assertIsInstance(service_code_config.callable_ref, str)
        self.assertIsNone(service_code_config._callable)
        self.assertIsNone(service_code_config.inline_code)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertTrue(service_code_config.file_set.is_remote())
        self.assertEqual(url, service_code_config.file_set.path)
        self.assertEqual(callable_ref, service_code_config.callable_ref)

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
