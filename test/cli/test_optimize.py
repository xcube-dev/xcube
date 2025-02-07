# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from test.cli.helpers import CliDataTest
from test.core.test_optimize import (
    INPUT_CUBE_FILE_SET,
    INPUT_CUBE_PATH,
    TEST_CUBE,
    list_file_set,
)

from xcube.core.dsio import rimraf

OUTPUT_CUBE_OPTIMIZED_DEFAULT_PATH = "test-optimized.zarr"
OUTPUT_CUBE_OPTIMIZED_USER_PATH = "fast-test.zarr"


class OptimizeDataTest(CliDataTest):
    def _clear_outputs(self):
        rimraf(INPUT_CUBE_PATH)
        rimraf(OUTPUT_CUBE_OPTIMIZED_DEFAULT_PATH)
        rimraf(OUTPUT_CUBE_OPTIMIZED_USER_PATH)

    def setUp(self):
        self._clear_outputs()
        TEST_CUBE.to_zarr(INPUT_CUBE_PATH)

    def tearDown(self):
        self._clear_outputs()

    def test_defaults(self):
        result = self.invoke_cli(["optimize", INPUT_CUBE_PATH])
        self.assertEqual(0, result.exit_code)

        expected_files = set(INPUT_CUBE_FILE_SET)
        expected_files.add(".zmetadata")
        self.assertEqual(
            expected_files, list_file_set(OUTPUT_CUBE_OPTIMIZED_DEFAULT_PATH)
        )

    def test_user_output(self):
        result = self.invoke_cli(
            ["optimize", "-o", OUTPUT_CUBE_OPTIMIZED_USER_PATH, "-C", INPUT_CUBE_PATH]
        )
        self.assertEqual(0, result.exit_code)

        expected_files = set(INPUT_CUBE_FILE_SET)
        expected_files.add(".zmetadata")
        expected_files.remove("time/1")
        expected_files.remove("time/2")
        expected_files.remove("time_bnds/1.0")
        expected_files.remove("time_bnds/2.0")
        self.assertEqual(expected_files, list_file_set(OUTPUT_CUBE_OPTIMIZED_USER_PATH))

    def test_in_place(self):
        result = self.invoke_cli(["optimize", "-IC", INPUT_CUBE_PATH])
        self.assertEqual(0, result.exit_code)

        expected_files = set(INPUT_CUBE_FILE_SET)
        expected_files.add(".zmetadata")
        expected_files.remove("time/1")
        expected_files.remove("time/2")
        expected_files.remove("time_bnds/1.0")
        expected_files.remove("time_bnds/2.0")
        self.assertEqual(expected_files, list_file_set(INPUT_CUBE_PATH))


class OptimizeTest(CliDataTest):
    def test_help_option(self):
        result = self.invoke_cli(["optimize", "--help"])
        self.assertEqual(0, result.exit_code)
