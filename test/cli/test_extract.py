import os
import os.path

from test.cli.test_cli import CliDataTest


class ExtractCliTest(CliDataTest):

    def test_help_option(self):
        result = self.invoke_cli(['extract', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_extract_points(self):
        base_dir = os.path.dirname(__file__)
        result = self.invoke_cli(['extract',
                                  os.path.join(base_dir,
                                               '..', '..', 'xcube', 'webapi', 'res', 'demo', 'cube.nc'),
                                  os.path.join(base_dir,
                                               'extract-points', 'points.csv'),
                                  ])
        self.assertEqual(0, result.exit_code)
        expected_output = ('idx,quality_flags,kd489,conc_tsm,conc_chl,c2rcc_flags,id_ref,lat_ref,lon_ref,time_ref\n'
                           '0,,,,,,0,51.75,2.4,2017-01-16T10:00:00Z\n'
                           '1,,,,,,1,51.75,2.5,2017-01-16T10:00:00Z\n'
                           '2,2290089984.0,,,,,2,51.75,2.6,2017-01-28T10:00:00Z\n'
                           '3,2147483648.0,,,,,3,51.75,2.7,2017-01-28T10:00:00Z\n')
        actual_output = result.stdout.replace('\r', '')
        self.assertEqual(len(expected_output), len(actual_output))
        self.assertEqual(expected_output, actual_output)
