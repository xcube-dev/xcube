import os
import os.path
import warnings
from io import StringIO

import pandas as pd

from test.cli.helpers import CliDataTest

VAR_COLS = [
    'c2rcc_flags',
    'conc_chl',
    'conc_tsm',
    'kd489',
    'quality_flags',
]

COORDS_COLS = [
    'lon',
    'lat',
    'time',
]

BOUNDS_COLS = [
    'lon_lower',
    'lon_upper',
    'lat_lower',
    'lat_upper',
    'time_lower',
    'time_upper',
]

INDEX_COLS = [
    'lon_index',
    'lat_index',
    'time_index',
]

REF_COLS = [
    'id_ref',
    'lon_ref',
    'lat_ref',
    'time_ref',
]


class ExtractCliTest(CliDataTest):

    @classmethod
    def setUpClass(cls):
        # We parse CSV-output from CLI using pandas in these tests.
        # Warnings will cause the pandas parser to fail.
        warnings.filterwarnings("ignore")

    @classmethod
    def tearDownClass(cls):
        warnings.resetwarnings()

    def test_help_option(self):
        result = self.invoke_cli(['extract', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_all_invalid_points(self):
        self._do_test_points('all-invalid-points.csv')

    def test_some_invalid_points(self):
        self._do_test_points('some-invalid-points.csv')

    def test_extract_points_with_refs(self):
        self._do_test_points('all-valid-points.csv')

    def _do_test_points(self, filename):
        self._do_test_points_with_options(filename, options=[],
                                          expected_length=6)
        self._do_test_points_with_options(filename, options=['--coords'],
                                          expected_length=6)
        self._do_test_points_with_options(filename, options=['--bounds'],
                                          expected_length=6)
        self._do_test_points_with_options(filename, options=['--refs'],
                                          expected_length=6)
        self._do_test_points_with_options(filename, options=['--indexes'],
                                          expected_length=6)
        self._do_test_points_with_options(filename, options=['--coords', '--bounds', '--refs', '--indexes'],
                                          expected_length=6)

    def _do_test_points_with_options(self,
                                     filename,
                                     options=None,
                                     expected_length=0):
        base_dir = os.path.dirname(__file__)
        args = ['extract']
        if options:
            args += options
        args += [os.path.join(base_dir, '..', '..', 'examples', 'serve', 'demo', 'cube.nc'),
                 os.path.join(base_dir, 'extract-points', filename)]
        result = self.invoke_cli(args)
        self.assertEqual(0, result.exit_code)
        self.assertNotEqual('', result.stdout)
        try:
            df = pd.read_csv(StringIO(result.stdout),
                             index_col='idx')
        except pd.errors.ParserError as e:
            print(f"Failed to parse result: {e}:")
            print(result.stdout)
            raise e
        self.assertEqual(expected_length, len(df))
        expected_columns = set(VAR_COLS)
        if '--coords' in options:
            expected_columns.update(COORDS_COLS)
        if '--bounds' in options:
            expected_columns.update(BOUNDS_COLS)
        if '--refs' in options:
            expected_columns.update(REF_COLS)
        if '--indexes' in options:
            expected_columns.update(INDEX_COLS)
        self.assertEqual(expected_columns, set(df.columns))
