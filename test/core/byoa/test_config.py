import os.path
import unittest

from xcube.core.byoa import CodeConfig
from xcube.core.byoa import FileSet

INLINE_CODE = (
    'def process_dataset(ds, text="Hello"):\n'
    '    return ds.assign_attrs(comment=f"xcube says {text}")\n'
)

PARENT_DIR = os.path.dirname(__file__)
LOCAL_MODULE_DIR = os.path.join(PARENT_DIR,
                                'test_data',
                                'user_code')
LOCAL_MODULE_ZIP = os.path.join(PARENT_DIR,
                                'test_data',
                                'user_code.zip')
LOCAL_PREFIXED_MODULE_ZIP = os.path.join(PARENT_DIR,
                                         'test_data',
                                         'user_code_prefixed.zip')


class CodeConfigTest(unittest.TestCase):

    def test_from_code_string(self):
        code_config = CodeConfig.from_code(
            INLINE_CODE,
            module_name='user_code_1',
            callable_params={'text': 'good bye'}
        )
        self.assertIsInstance(code_config, CodeConfig)
        self.assertEqual(INLINE_CODE, code_config.inline_code)
        self.assertEqual({'text': 'good bye'}, code_config.callable_params)
        self.assertEqual('user_code_1:process_dataset',
                         code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_from_code_with_function_refs(self):
        code_config = CodeConfig.from_code(
            'import xarray as xr\n',
            modify_dataset,
            transform_dataset,
            module_name='user_code_2',
            callable_params={'text': 'good bye'}
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
        self.assertEqual({'text': 'good bye'}, code_config.callable_params)
        self.assertEqual('user_code_2:modify_dataset', code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_from_callable(self):
        code_config = CodeConfig.from_callable(
            modify_dataset,
            callable_params=dict(text='Good bye!')
        )
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsNone(code_config.inline_code)
        self.assertIsNone(code_config.file_set, FileSet)
        self.assertIsNone(code_config.callable_ref)
        self.assertIs(modify_dataset, code_config.get_callable())

    def test_from_file_set_dir(self):
        code_config = CodeConfig.from_file_set(
            LOCAL_MODULE_DIR,
            'processor:process_dataset'
        )
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsNone(code_config.inline_code)
        self.assertIsInstance(code_config.file_set, FileSet)
        self.assertEqual('processor:process_dataset', code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_from_file_set_zip(self):
        code_config = CodeConfig.from_file_set(
            LOCAL_MODULE_ZIP,
            'processor:process_dataset'
        )
        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsNone(code_config.inline_code)
        self.assertIsInstance(code_config.file_set, FileSet)
        self.assertEqual('processor:process_dataset',
                         code_config.callable_ref)
        self.assertTrue(callable(code_config.get_callable()))

    def test_from_github_release(self):
        code_config = CodeConfig.from_github_archive(
            'dcs4cop',
            'xcube-byoa-examples',
            'v0.1.0.dev0',
            '0.1.0.dev0',
            'xcube_byoa_1.processor:process_dataset'
        )

        self.assertIsInstance(code_config, CodeConfig)
        self.assertIsNone(code_config.inline_code)
        self.assertIsInstance(code_config.file_set, FileSet)
        self.assertEqual('https://github.com/dcs4cop/'
                         'xcube-byoa-examples/archive/v0.1.0.dev0.zip',
                         code_config.file_set.path)
        self.assertEqual('xcube_byoa_1.processor:process_dataset',
                         code_config.callable_ref)

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
                         'xcube-byoa-*.')
        self.assertRegex(local_code_config.callable_ref,
                         'xcube_byoa_*.:process_dataset')
        # CodeConfigs from for_local() shall be able to load callable
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_from_file_set_dir(self):
        user_code_config = CodeConfig.from_file_set(
            LOCAL_MODULE_DIR,
            'processor:process_dataset'
        )

        local_code_config = user_code_config.for_local()
        self.assertIs(user_code_config, local_code_config)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertIs(user_code_config.file_set, local_code_config.file_set)
        self.assertEqual(LOCAL_MODULE_DIR, local_code_config.file_set.path)
        # CodeConfigs from for_local() shall be able to load callable
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_from_zip_file_set(self):
        user_code_config = CodeConfig.from_file_set(
            LOCAL_MODULE_ZIP,
            'processor:process_dataset'
        )

        local_code_config = user_code_config.for_local()
        self.assertIs(user_code_config, local_code_config)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertIs(user_code_config.file_set, local_code_config.file_set)
        self.assertEqual(LOCAL_MODULE_ZIP, local_code_config.file_set.path)
        # CodeConfigs from for_local() shall be able to load callable
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_from_prefixed_zip_file_set(self):
        user_code_config = CodeConfig.from_file_set(
            FileSet(LOCAL_PREFIXED_MODULE_ZIP, sub_path='user_code'),
            'processor:process_dataset',
        )

        local_code_config = user_code_config.for_local()
        self.assertIsInstance(local_code_config, CodeConfig)
        self.assertIsInstance(local_code_config.callable_ref, str)
        self.assertIsNone(local_code_config._callable)
        self.assertIsNone(local_code_config.inline_code)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertTrue(local_code_config.file_set.is_local_dir())
        self.assertRegex(os.path.basename(local_code_config.file_set.path),
                         'xcube-byoa-*.')
        self.assertEqual('processor:process_dataset',
                         local_code_config.callable_ref)
        # CodeConfigs from for_local() shall be able to load callable
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_from_github_release(self):
        user_code_config = CodeConfig.from_github_archive(
            gh_org='dcs4cop',
            gh_repo='xcube-byoa-examples',
            gh_tag='v0.1.0.dev0',
            gh_release='0.1.0.dev0',
            callable_ref='xcube_byoa_ex1.processor:process_dataset',
        )

        local_code_config = user_code_config.for_local()
        self.assertIsInstance(local_code_config, CodeConfig)
        self.assertIsInstance(local_code_config.callable_ref, str)
        self.assertIsNone(local_code_config._callable)
        self.assertIsNone(local_code_config.inline_code)
        self.assertIsInstance(local_code_config.file_set, FileSet)
        self.assertTrue(local_code_config.file_set.is_local_dir())
        self.assertRegex(os.path.basename(local_code_config.file_set.path),
                         'xcube-byoa-*.')
        self.assertEqual('xcube_byoa_ex1.processor:process_dataset',
                         local_code_config.callable_ref)
        # CodeConfigs from for_local() shall be able to load callable
        self.assertTrue(callable(local_code_config.get_callable()))

    def test_for_local_illegal_state(self):
        code_config = CodeConfig.from_code(INLINE_CODE)
        code_config.inline_code = None
        with self.assertRaises(RuntimeError) as e:
            code_config.for_local()
        self.assertEqual(
            ('CodeConfig.for_local() failed'
             ' due to an invalid internal state',),
            e.exception.args
        )

    def test_for_service_from_callable(self):
        user_code_config = CodeConfig.from_callable(transform_dataset)

        service_code_config = user_code_config.for_service()
        self.assertIsInstance(service_code_config, CodeConfig)
        self.assertIsInstance(service_code_config.callable_ref, str)
        self.assertIsNone(service_code_config._callable)
        self.assertIsNone(service_code_config.inline_code)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertTrue(service_code_config.file_set.is_local_zip())

    def test_for_service_from_zip_file_set(self):
        user_code_config = CodeConfig.from_file_set(
            LOCAL_MODULE_ZIP,
            'processor:process_dataset'
        )

        service_code_config = user_code_config.for_service()
        self.assertIs(user_code_config, service_code_config)
        self.assertIs(user_code_config.file_set, service_code_config.file_set)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertEqual(LOCAL_MODULE_ZIP, service_code_config.file_set.path)

    def test_for_service_from_file_set_dir(self):
        user_code_config = CodeConfig.from_file_set(
            LOCAL_MODULE_DIR,
            'processor:process_dataset'
        )

        service_code_config = user_code_config.for_service()
        self.assertIsInstance(service_code_config, CodeConfig)
        self.assertIsInstance(service_code_config.callable_ref, str)
        self.assertIsNone(service_code_config._callable)
        self.assertIsNone(service_code_config.inline_code)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertTrue(service_code_config.file_set.is_local_zip())
        self.assertRegex(os.path.basename(service_code_config.file_set.path),
                         'xcube-byoa-*.')
        self.assertEqual('processor:process_dataset',
                         service_code_config.callable_ref)

    def test_for_service_from_github_release(self):
        user_code_config = CodeConfig.from_github_archive(
            gh_org='dcs4cop',
            gh_repo='xcube-byoa-examples',
            gh_tag='v0.1.0.dev0',
            gh_release='0.1.0.dev0',
            callable_ref='xcube_byoa_1.processor:process_dataset'
        )

        service_code_config = user_code_config.for_service()
        self.assertIsInstance(service_code_config, CodeConfig)
        self.assertIsInstance(service_code_config.callable_ref, str)
        self.assertIsNone(service_code_config._callable)
        self.assertIsNone(service_code_config.inline_code)
        self.assertIsInstance(service_code_config.file_set, FileSet)
        self.assertFalse(service_code_config.file_set.is_local_dir())
        self.assertEqual('https://github.com/dcs4cop/'
                         'xcube-byoa-examples/archive/v0.1.0.dev0.zip',
                         service_code_config.file_set.path)
        self.assertEqual('xcube_byoa_1.processor:process_dataset',
                         service_code_config.callable_ref)

    def test_for_service_illegal_state(self):
        code_config = CodeConfig.from_code(INLINE_CODE)
        code_config.inline_code = None
        with self.assertRaises(RuntimeError) as e:
            code_config.for_service()
        self.assertEqual(
            ('for_service() failed due to an invalid CodeConfig state',),
            e.exception.args
        )

    def test_to_dict(self):
        d = CodeConfig.from_callable(modify_dataset).to_dict()
        # Shall we raise instead?
        self.assertEqual({}, d)

        d = CodeConfig.from_code(
            INLINE_CODE, module_name='user_code'
        ).to_dict()
        self.assertEqual(
            {
                'callable_ref': 'user_code:process_dataset',
                'inline_code': INLINE_CODE,
            },
            d
        )

        d = CodeConfig.from_file_set(
            FileSet('github://dcs4cop:xcube@v0.8.2.dev0',
                    includes='*.py'),
            callable_ref=('test.core.byoa.test_config:'
                          'modify_dataset')
        ).to_dict()
        self.assertIsInstance(d.get('file_set'), dict)
        self.assertEqual(('test.core.byoa.test_config:'
                          'modify_dataset'),
                         d.get('callable_ref'))


def modify_dataset(ds, text="Hello"):
    return transform_dataset(ds, text)


def transform_dataset(ds, text):
    return ds.assign_attrs(comment=f"xcube says {text}")
