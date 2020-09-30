import os
import unittest

from test.cli.helpers import CliDataTest


@unittest.skip('Very slow tests, execute on demand only')
class BenchmarkCliTest(CliDataTest):

    def test_help_option(self):
        result = self.invoke_cli(['benchmark', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_simple(self):
        config_path = os.path.join(os.path.dirname(__file__), 'benchmark-configs', 'simple.yml')
        result = self.invoke_cli(['benchmark', config_path])
        msg = f'actual output:\n{result.stdout}'
        self.assertTrue('# command template: xcube dump ${input}\n' in result.stdout, msg=msg)
        self.assertTrue('\n# repetition count: 1\n' in result.stdout, msg=msg)
        self.assertTrue('\nid;input;time\n' in result.stdout, msg=msg)
        self.assertTrue('\n0;test.zarr;' in result.stdout, msg=msg)
        self.assertTrue('\n1;test.nc;' in result.stdout, msg=msg)

    def test_simple_with_repetitions(self):
        config_path = os.path.join(os.path.dirname(__file__), 'benchmark-configs', 'simple.yml')
        result = self.invoke_cli(['benchmark', '--repeats', 3, config_path])
        msg = f'actual output:\n{result.stdout}'

        self.assertTrue('# command template: xcube dump ${input}\n' in result.stdout, msg=msg)
        self.assertTrue('\n# repetition count: 3\n' in result.stdout, msg=msg)
        self.assertTrue('\nid;input;time-mean;time-median;time-stdev;time-min;time-max\n' in result.stdout, msg=msg)
        self.assertTrue('\n0;test.zarr;' in result.stdout, msg=msg)
        self.assertTrue('\n1;test.nc;' in result.stdout, msg=msg)
