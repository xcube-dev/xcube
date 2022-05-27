import json
import os.path

import yaml

from test.cli.helpers import CliTest
from xcube.core.dsio import rimraf


class IOTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['io', '--help'])
        self.assertEqual(0, result.exit_code)


class IOStoreTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['io', 'store', '--help'])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(['io', 'store', 'list', '--help'])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(['io', 'store', 'info', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_list(self):
        result = self.invoke_cli(['io', 'store', 'list'])
        self.assertEqual(0, result.exit_code)

    def test_info(self):
        result = self.invoke_cli(['io', 'store', 'info', 'memory',
                                  '-POWj'])
        self.assertEqual(0, result.exit_code)
        result = self.invoke_cli(['io', 'store', 'info', 'file',
                                  '-POW', 'root=.'])
        self.assertEqual(0, result.exit_code)

        # param "root" missing
        result = self.invoke_cli(['io', 'store', 'info',
                                  'file', '-POW', 'root=42'])
        self.assertEqual(1, result.exit_code)
        self.assertEqual("Error: Failed to instantiate data store 'file'"
                         " with parameters ('root=42',):"
                         " Invalid parameterization detected:"
                         " 42 is not of type 'string'\n",
                         result.stderr)

    def test_data(self):
        demo_dir = os.path.join(os.path.dirname(__file__),
                                '..', '..',
                                'examples', 'serve', 'demo')

        result = self.invoke_cli(['io', 'store', 'info',
                                  'file', '-D', f'root={demo_dir}'])
        self.assertEqual(0, result.exit_code)
        self.assertEqual(('\n'
                          'Data store description:\n'
                          '  Data store that uses a local filesystem\n'
                          '\n'
                          'Data resources:\n'
                          '             cube-1-250-250.levels  <no title>\n'
                          '               cube-1-250-250.zarr  <no title>\n'
                          '               cube-5-100-200.zarr  <no title>\n'
                          '                           cube.nc  <no title>\n'
                          '                    sample-cog.tif  <no title>\n'
                          '                sample-geotiff.tif  <no title>\n'
                          '6 data resources found.\n'),
                         result.stdout)
        demo_dir = os.path.join(os.path.dirname(__file__),
                                '..', '..',
                                'examples', 'serve', 'demo')
        result = self.invoke_cli(['io', 'store', 'data',
                                  'file', 'cube-1-250-250.zarr',
                                  f'root={demo_dir}'])
        self.assertEqual(0, result.exit_code)
        self.assertIn('cube-1-250-250.zarr', result.stdout)


class IOOpenerTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['io', 'opener', '--help'])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(['io', 'opener', 'list', '--help'])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(['io', 'opener', 'info', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_list(self):
        result = self.invoke_cli(['io', 'opener', 'list'])
        self.assertEqual(0, result.exit_code)

    def test_info(self):
        result = self.invoke_cli(['io', 'opener', 'info', 'dataset:zarr:file'])
        self.assertEqual(0, result.exit_code)


class IOWriterTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['io', 'writer', '--help'])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(['io', 'writer', 'list', '--help'])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(['io', 'writer', 'info', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_list(self):
        result = self.invoke_cli(['io', 'writer', 'list'])
        self.assertEqual(0, result.exit_code)

    def test_info(self):
        result = self.invoke_cli(['io', 'writer', 'info', 'dataset:zarr:file'])
        self.assertEqual(0, result.exit_code)


class IODumpTest(CliTest):
    STORE_CONF = {
        "this_dir": {
            "store_id": "file",
            "store_params": {
                "root": os.path.join(os.path.dirname(__file__),
                                     "..", "..",
                                     "examples", "serve", "demo")
            }
        }
    }

    @classmethod
    def setUpClass(cls) -> None:
        with open('store-conf.yml', 'w') as fp:
            yaml.dump(cls.STORE_CONF, fp)
        with open('store-conf.json', 'w') as fp:
            json.dump(cls.STORE_CONF, fp)

    @classmethod
    def tearDownClass(cls) -> None:
        rimraf('store-conf.json')
        rimraf('store-conf.yml')

    def tearDown(self) -> None:
        rimraf('store-dump.json')
        rimraf('store-dump.yml')
        rimraf('store-dump.csv')
        rimraf('out.json')
        rimraf('out.jml')
        rimraf('out.txt')

    def test_help_option(self):
        result = self.invoke_cli(['io', 'dump',
                                  '--help'])
        self.assertEqual(0, result.exit_code)

    def test_yaml_config(self):
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.yml'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('store-dump.json'))

    def test_yaml_config_short_format(self):
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.yml',
                                  '--short', '--json'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('store-dump.json'))
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.yml',
                                  '--short', '--yaml'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('store-dump.yml'))
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.yml',
                                  '--short', '--csv'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('store-dump.csv'))

    def test_json_config_short(self):
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.yml',
                                  '--short'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('store-dump.json'))

    def test_json_config_output(self):
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.json',
                                  '-o', 'out.json'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('out.json'))
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.json',
                                  '-o', 'out.yml'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('out.yml'))
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.json',
                                  '-o', 'out.txt'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('out.txt'))

    def test_json_config_type(self):
        result = self.invoke_cli(['io', 'dump',
                                  '-c', 'store-conf.json',
                                  '-t', 'dataset'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('store-dump.json'))
