import os.path

from test.cli.helpers import CliTest


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
                         '  base_dir 	(string)\n',
                         result.stdout)

    def test_data(self):
        demo_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'serve', 'demo')

        result = self.invoke_cli(['io', 'store', 'info', 'directory', '-D', f'base_dir={demo_dir}'])
        self.assertEqual(0, result.exit_code)
        self.assertEqual('Directory data store\n'
                         'Data resources:\n'
                         '  cube-1-250-250.zarr\n'
                         '  cube-5-100-200.zarr\n'
                         '  cube.nc\n'
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
