import os.path
from typing import List, Tuple

from test.cli.helpers import CliDataTest
from test.cli.helpers import CliTest
from test.cli.helpers import TEST_NC_FILE, TEST_ZARR_DIR
from xcube.core.level import read_levels
from xcube.core.verify import assert_cube


class LevelTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['level', '--help'])
        self.assertEqual(0, result.exit_code)


class LevelDataTest(CliDataTest):
    TEST_OUTPUT = "test.levels"

    def outputs(self) -> List[str]:
        return [LevelDataTest.TEST_OUTPUT, 'my.levels']

    def test_all_defaults(self):
        result = self.invoke_cli(['level', TEST_ZARR_DIR])
        self._assert_result_ok(
            result, [((1, 1, 1, 1, 1), (90, 90), (180, 180)),
                     ((1, 1, 1, 1, 1), (90,), (180,))],
            LevelDataTest.TEST_OUTPUT,
            'Level 1 of 2 written after .*\n'
            'Level 2 of 2 written after .*\n'
            '2 level\(s\) written into test.levels after .*\n'
        )

    def test_with_output(self):
        result = self.invoke_cli(['level', TEST_ZARR_DIR,
                                  '--output', 'my.levels'])
        self._assert_result_ok(
            result, [((1, 1, 1, 1, 1), (90, 90), (180, 180)),
                     ((1, 1, 1, 1, 1), (90,), (180,))], 'my.levels',
            'Level 1 of 2 written after .*\n'
            'Level 2 of 2 written after .*\n'
            '2 level\(s\) written into my.levels after .*\n'
        )

    def test_with_tile_size_and_num_levels(self):
        result = self.invoke_cli(['level', TEST_ZARR_DIR,
                                  '-t', '90,45', '-n', '4'])
        self._assert_result_ok(
            result, [((1, 1, 1, 1, 1), (45, 45, 45, 45), (90, 90, 90, 90)),
                     ((1, 1, 1, 1, 1), (45, 45), (90, 90)),
                     ((1, 1, 1, 1, 1), (45,), (90,))],
            LevelDataTest.TEST_OUTPUT,
            'Level 1 of 3 written after .*\n'
            'Level 2 of 3 written after .*\n'
            'Level 3 of 3 written after .*\n'
            '3 level\(s\) written into test.levels after .*\n'
        )

    def _assert_result_ok(self,
                          result,
                          level_chunks: List[Tuple],
                          output_path: str,
                          message_regex: str):
        self.assertEqual(0, result.exit_code)
        self.assertRegex(result.stdout, message_regex)
        self.assertTrue(os.path.isdir(output_path))
        level_datasets = read_levels(output_path)
        level = 0
        for level_dataset in level_datasets:
            assert_cube(level_dataset)
            self.assertEqual({'precipitation',
                              'soil_moisture',
                              'temperature'},
                             set(level_dataset.data_vars.keys()))
            for var_name, var in level_dataset.data_vars.items():
                var_chunks = level_chunks[level]
                self.assertEqual(var_chunks,
                                 var.chunks,
                                 f'{var_name} at level {level}')
            level += 1

    def _assert_result_not_ok(self, result, message_regex: str):
        self.assertEqual(1, result.exit_code)
        self.assertRegex(result.stdout, message_regex)

    def test_level_with_nc(self):
        result = self.invoke_cli(["level",
                                  "-t", "45",
                                  "-o", LevelDataTest.TEST_OUTPUT,
                                  TEST_NC_FILE,
                                  ])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual({'0.zarr', '1.zarr', '2.zarr'},
                         set(os.listdir(LevelDataTest.TEST_OUTPUT)))

    def test_level_with_zarr(self):
        result = self.invoke_cli(["level",
                                  "-t", "45",
                                  "-o", LevelDataTest.TEST_OUTPUT,
                                  TEST_ZARR_DIR,
                                  ])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual({'0.zarr', '1.zarr', '2.zarr'},
                         set(os.listdir(LevelDataTest.TEST_OUTPUT)))

    def test_level_with_zarr_link(self):
        result = self.invoke_cli(["level",
                                  "--link",
                                  "-t", "45",
                                  "-o", LevelDataTest.TEST_OUTPUT,
                                  TEST_ZARR_DIR,
                                  ])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual({'0.link', '1.zarr', '2.zarr'},
                         set(os.listdir(LevelDataTest.TEST_OUTPUT)))

    def test_level_with_zarr_num_levels_max(self):
        result = self.invoke_cli(["level",
                                  "-t", "45",
                                  "-n", "2",
                                  "-o", LevelDataTest.TEST_OUTPUT,
                                  TEST_ZARR_DIR,
                                  ])
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual({'0.zarr', '1.zarr'},
                         set(os.listdir(LevelDataTest.TEST_OUTPUT)))

    def test_invalid_inputs(self):
        result = self.invoke_cli(["level",
                                  "-t", "a45",
                                  "-o", LevelDataTest.TEST_OUTPUT,
                                  TEST_NC_FILE,
                                  ])
        self._assert_result_not_ok(
            result,
            "Error\\: Invalid tile sizes in TILE_SIZE found: "
            "invalid literal for int\\(\\) with base 10\\: 'a45'\n"
        )

        result = self.invoke_cli(["level",
                                  "-t", "-3",
                                  "-o", LevelDataTest.TEST_OUTPUT,
                                  TEST_NC_FILE,
                                  ])
        self._assert_result_not_ok(
            result,
            "Error\\: Invalid tile sizes in TILE_SIZE found\\: "
            "all items must be positive integer numbers\n"
        )

        result = self.invoke_cli(["level",
                                  "-t", "45,45,45",
                                  "-o", LevelDataTest.TEST_OUTPUT,
                                  TEST_NC_FILE])
        self._assert_result_not_ok(
            result,
            "Error\\: TILE_SIZE must have 2 tile sizes separated by ','\n"
        )

        result = self.invoke_cli(["level",
                                  "-n", "0",
                                  "-o", LevelDataTest.TEST_OUTPUT,
                                  TEST_NC_FILE])
        self._assert_result_not_ok(
            result,
            "NUM_LEVELS_MAX must be a positive integer\n"
        )

    def test_with_existing_output(self):
        result = self.invoke_cli(['level', TEST_ZARR_DIR,
                                  '--output', 'my.levels'])
        result = self.invoke_cli(['level', TEST_ZARR_DIR,
                                  '--output', 'my.levels'])
        self._assert_result_not_ok(
            result,
            'Error: output \'my\\.levels\' already exists\n'
        )
