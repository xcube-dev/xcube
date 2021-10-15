import unittest

import xarray as xr

# noinspection PyUnresolvedReferences
import xcube.core.xarray
from xcube.core.byoa import CodeConfig
from xcube.core.gen2 import CubeGeneratorError, CubeConfig
from xcube.core.gen2.local.usercode import CubeUserCodeExecutor
from xcube.core.gen2.processor import DatasetProcessor
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


def process_dataset_function(dataset: xr.Dataset,
                             name: str = None, value: int = None):
    dataset = dataset.copy()
    return dataset.assign(**{name: value})


class GoodDuckProcessor:
    @classmethod
    def get_process_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                name=JsonStringSchema(min_length=1),
                value=JsonIntegerSchema(minimum=1),
            ),
            required=['name', 'value'],
            additional_properties=False,
        )

    # noinspection PyMethodMayBeStatic
    def process_dataset(self, dataset: xr.Dataset,
                        name: str = None, value: int = None) \
            -> xr.Dataset:
        return process_dataset_function(dataset, name=name, value=value)


class GoodProcessor(GoodDuckProcessor, DatasetProcessor):
    pass


class BadSchemaProcessor(GoodProcessor):
    @classmethod
    def get_process_params_schema(cls) -> JsonObjectSchema:
        # noinspection PyTypeChecker
        return JsonIntegerSchema()


class NoProcessProcessor:
    pass


class BadProcessProcessor:
    process_dataset = True


class CubeUserCodeExecutorTest(unittest.TestCase):
    good_params = dict(name='X', value=42)
    bad_params = dict(name='X', value=0)

    def test_function(self):
        self.assertCallableWorks(process_dataset_function)

    def test_class_duck_typed(self):
        self.assertCallableWorks(GoodDuckProcessor)

    def test_class_strongly_typed(self):
        self.assertCallableWorks(GoodProcessor)

    def assertCallableWorks(self, user_code_callable):
        code_config = CodeConfig(_callable=user_code_callable,
                                 callable_params=self.good_params)
        executor = CubeUserCodeExecutor(code_config)
        ds_input = new_cube(variables=dict(a=1))
        ds_output, gm, cc = executor.transform_cube(
            ds_input,
            GridMapping.from_dataset(ds_input),
            CubeConfig()
        )
        self.assertIsInstance(ds_output, xr.Dataset)
        self.assertIsInstance(gm, GridMapping)
        self.assertIsInstance(cc, CubeConfig)
        self.assertIsNot(ds_output, ds_input)
        self.assertIn('X', ds_output)
        self.assertEqual(42, ds_output.X)

    def test_class_bad_params(self):
        code_config = CodeConfig(_callable=GoodProcessor,
                                 callable_params=self.bad_params)
        with self.assertRaises(CubeGeneratorError) as ctx:
            CubeUserCodeExecutor(code_config)
        self.assertEqual(400, ctx.exception.status_code)
        self.assertEqual("Invalid processing parameters:"
                         " 0 is less than the minimum of 1\n"
                         "\n"
                         "Failed validating 'minimum'"
                         " in schema['properties']['value']:\n"
                         "    {'minimum': 1, 'type': 'integer'}\n"
                         "\n"
                         "On instance['value']:\n"
                         "    0",
                         f'{ctx.exception}')

    def test_class_bad_schema(self):
        code_config = CodeConfig(_callable=BadSchemaProcessor,
                                 callable_params=self.good_params)
        with self.assertRaises(CubeGeneratorError) as ctx:
            CubeUserCodeExecutor(code_config)
        self.assertEqual(400, ctx.exception.status_code)
        self.assertEqual(f"Parameter schema returned by"
                         f" user code class"
                         f" {BadSchemaProcessor!r}"
                         f" must be an instance of"
                         f" {JsonObjectSchema!r}",
                         f'{ctx.exception}')

    def test_class_no_process(self):
        code_config = CodeConfig(_callable=NoProcessProcessor,
                                 callable_params=self.good_params)
        with self.assertRaises(CubeGeneratorError) as ctx:
            CubeUserCodeExecutor(code_config)
        self.assertEqual(400, ctx.exception.status_code)
        self.assertEqual(f"Missing method 'process_dataset'"
                         f" in user code class"
                         f" {NoProcessProcessor!r}",
                         f'{ctx.exception}')

    def test_class_bad_process(self):
        code_config = CodeConfig(_callable=BadProcessProcessor,
                                 callable_params=self.good_params)
        with self.assertRaises(CubeGeneratorError) as ctx:
            CubeUserCodeExecutor(code_config)
        self.assertEqual(400, ctx.exception.status_code)
        self.assertEqual(f"Attribute 'process_dataset'"
                         f" of user code class"
                         f" {BadProcessProcessor!r}"
                         f" must be callable",
                         f'{ctx.exception}')
