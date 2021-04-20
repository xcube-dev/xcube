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
        result = self.invoke_cli(['io', 'store', 'info', 'memory', '-POWj'])
        self.assertEqual(0, result.exit_code)
        result = self.invoke_cli(['io', 'store', 'info', 'directory', '-POW', 'base_dir=.'])
        self.assertEqual(0, result.exit_code)

        # param "base_dir" missing
        result = self.invoke_cli(['io', 'store', 'info', 'directory', '-POW'])
        self.assertEqual(1, result.exit_code)
        self.assertEqual('Error: Data store "directory" has required parameters, but none were given.\n'
                         'Required parameters:\n'
                         '                  base_dir  (string)\n',
                         result.stdout)

    def test_data(self):
        demo_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'serve', 'demo')

        result = self.invoke_cli(['io', 'store', 'info', 'directory', '-D', f'base_dir={demo_dir}'])
        self.assertEqual(0, result.exit_code)
        self.assertEqual('\n'
                         'Data store description:\n'
                         '  Directory data store\n'
                         '\n'
                         'Data resources:\n'
                         '               cube-1-250-250.zarr  <no title>\n'
                         '               cube-5-100-200.zarr  <no title>\n'
                         '                           cube.nc  <no title>\n'
                         '3 data resources found.\n',
                         result.stdout)

        demo_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'serve', 'demo')
        result = self.invoke_cli(['io', 'store', 'data', 'directory', 'cube-1-250-250.zarr', f'base_dir={demo_dir}'])
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
        result = self.invoke_cli(['io', 'opener', 'info', 'dataset:zarr:posix'])
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
        result = self.invoke_cli(['io', 'writer', 'info', 'dataset:zarr:posix'])
        self.assertEqual(0, result.exit_code)


class IODumpTest(CliTest):
    STORE_CONF = {
        "this_dir": {
            "store_id": "directory",
            "store_params": {
                "base_dir": "."
            }
        }
    }

    def setUp(self) -> None:
        with open('store-conf.yml', 'w') as fp:
            yaml.dump(self.STORE_CONF, fp)
        with open('store-conf.json', 'w') as fp:
            json.dump(self.STORE_CONF, fp)

    def tearDown(self) -> None:
        rimraf('store-dump.json')
        rimraf('store-conf.yml')
        rimraf('out.json')

    def test_help_option(self):
        result = self.invoke_cli(['io', 'dump', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_yaml_config(self):
        result = self.invoke_cli(['io', 'dump', '-c', 'store-conf.yml'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('store-dump.json'))

    def test_json_config_output(self):
        result = self.invoke_cli(['io', 'dump', '-c', 'store-conf.json', '-o', 'out.json'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('out.json'))

    def test_json_config_type(self):
        result = self.invoke_cli(['io', 'dump', '-c', 'store-conf.json', '-t', 'dataset[cube]'])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.exists('store-dump.json'))
