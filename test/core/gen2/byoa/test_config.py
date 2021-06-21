import os.path
import unittest
import inspect

from xcube.core.gen2.byoa.config import CodeConfig


class CodeConfigTest(unittest.TestCase):

    def test_from_code_code_string(self):
        code_string = (
            'def process_dataset(ds, text="Hello"):\n'
            '    return ds.assign_attrs(comment=f"xcube says {text}")\n'
        )
        code_config = CodeConfig.from_code(
            code_string,
            parameters={'text': 'good bye'}
        )
        self.assertIsInstance(code_config, CodeConfig)
        self.assertEqual(code_string, code_config.code_string)
        self.assertEqual({'text': 'good bye'}, code_config.parameters)
        self.assertEqual('process_dataset', code_config.callable_name)

    def test_inspect(self):
        from test.core.gen2.byoa.user_code.processor import process_dataset
        self.assertEqual('process_dataset', process_dataset.__name__)
        self.assertEqual(os.path.join(os.path.dirname(__file__), 'user_code', 'processor.py'),
                         inspect.getfile(process_dataset))
        self.assertEqual('test.core.gen2.byoa.user_code.processor',
                         inspect.getmodule(process_dataset).__name__)

    def test_from_code_function_ref(self):
        code_config = CodeConfig.from_code(
            'import xarray as xr\n',
            modify_dataset,
            transform_dataset,
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
            code_config.code_string)
        self.assertEqual({'text': 'good bye'}, code_config.parameters)
        self.assertEqual('modify_dataset', code_config.callable_name)


def modify_dataset(ds, text="Hello"):
    return transform_dataset(ds, text)


def transform_dataset(ds, text):
    return ds.assign_attrs(comment=f"xcube says {text}")
