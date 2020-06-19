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
