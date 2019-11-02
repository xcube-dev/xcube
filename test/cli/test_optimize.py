from test.cli.test_cli import CliDataTest
from test.core.test_optimize import TEST_CUBE, TEST_CUBE_ZARR, TEST_CUBE_FILE_SET, list_file_set
from xcube.core.dsio import rimraf

TEST_CUBE_ZARR_OPTIMIZED_DEFAULT = 'test-optimized.zarr'
TEST_CUBE_ZARR_OPTIMIZED_USER = 'fast-test.zarr'


class OptimizeDataTest(CliDataTest):
    def _clear_outputs(self):
        rimraf(TEST_CUBE_ZARR)
        rimraf(TEST_CUBE_ZARR_OPTIMIZED_DEFAULT)
        rimraf(TEST_CUBE_ZARR_OPTIMIZED_USER)

    def setUp(self):
        self._clear_outputs()
        TEST_CUBE.to_zarr(TEST_CUBE_ZARR)

    def tearDown(self):
        self._clear_outputs()

    def test_defaults(self):
        result = self.invoke_cli(['optimize', TEST_CUBE_ZARR])
        self.assertEqual(0, result.exit_code)

        expected_files = set(TEST_CUBE_FILE_SET)
        expected_files.add('.zmetadata')
        self.assertEqual(expected_files, list_file_set(TEST_CUBE_ZARR_OPTIMIZED_DEFAULT))

    def test_user_output(self):
        result = self.invoke_cli(['optimize', '-o', TEST_CUBE_ZARR_OPTIMIZED_USER, '-C', TEST_CUBE_ZARR])
        self.assertEqual(0, result.exit_code)

        expected_files = set(TEST_CUBE_FILE_SET)
        expected_files.add('.zmetadata')
        expected_files.remove('time/1')
        expected_files.remove('time/2')
        expected_files.remove('time_bnds/1.0')
        expected_files.remove('time_bnds/2.0')
        self.assertEqual(expected_files, list_file_set(TEST_CUBE_ZARR_OPTIMIZED_USER))

    def test_in_place(self):
        result = self.invoke_cli(['optimize', '-IC', TEST_CUBE_ZARR])
        self.assertEqual(0, result.exit_code)

        expected_files = set(TEST_CUBE_FILE_SET)
        expected_files.add('.zmetadata')
        expected_files.remove('time/1')
        expected_files.remove('time/2')
        expected_files.remove('time_bnds/1.0')
        expected_files.remove('time_bnds/2.0')
        self.assertEqual(expected_files, list_file_set(TEST_CUBE_ZARR))


class OptimizeTest(CliDataTest):

    def test_help_option(self):
        result = self.invoke_cli(['optimize', '--help'])
        self.assertEqual(0, result.exit_code)
