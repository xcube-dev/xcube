# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os.path
from test.cli.helpers import TEST_NC_FILE, TEST_ZARR_DIR, CliDataTest, CliTest
from typing import Dict, List, Optional, Tuple

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import new_fs_data_store


class LevelTest(CliTest):
    def test_help_option(self):
        result = self.invoke_cli(["level", "--help"])
        self.assertEqual(0, result.exit_code)


class LevelDataTest(CliDataTest):
    TEST_OUTPUT = "test.levels"

    def outputs(self) -> list[str]:
        return [LevelDataTest.TEST_OUTPUT, "my.levels"]

    def chunks(self) -> Optional[dict[str, int]]:
        return dict(time=1, lat=90, lon=180)

    def test_all_defaults(self):
        result = self.invoke_cli(["level", "-v", TEST_ZARR_DIR])
        self._assert_result_ok(
            result,
            [((1, 1, 1, 1, 1), (90, 90), (180, 180)), ((1, 1, 1, 1, 1), (90,), (180,))],
            LevelDataTest.TEST_OUTPUT,
            "Multi-level dataset written to /test.levels after .*\n",
        )

    def test_with_output(self):
        result = self.invoke_cli(
            ["level", "-v", TEST_ZARR_DIR, "--output", "my.levels"]
        )
        self._assert_result_ok(
            result,
            [((1, 1, 1, 1, 1), (90, 90), (180, 180)), ((1, 1, 1, 1, 1), (90,), (180,))],
            "my.levels",
            "Multi-level dataset written to my.levels after .*\n",
        )

    def test_with_tile_size_and_num_levels(self):
        result = self.invoke_cli(
            ["level", "-v", TEST_ZARR_DIR, "-t", "90,45", "-n", "4"]
        )
        self._assert_result_ok(
            result,
            [
                ((1, 1, 1, 1, 1), (45, 45, 45, 45), (90, 90, 90, 90)),
                ((1, 1, 1, 1, 1), (45, 45), (90, 90)),
                ((1, 1, 1, 1, 1), (45,), (90,)),
                ((1, 1, 1, 1, 1), (23,), (45,)),
            ],
            LevelDataTest.TEST_OUTPUT,
            "Multi-level dataset written to /test.levels after .*\n",
        )

    def _assert_result_ok(
        self, result, level_chunks: list[tuple], output_path: str, message_regex: str
    ):
        self.assertEqual(0, result.exit_code)
        self.assertRegex(result.stderr, message_regex)
        self.assertTrue(os.path.isdir(output_path))
        data_store = new_fs_data_store("file")
        ml_dataset = data_store.open_data(output_path)
        self.assertIsInstance(ml_dataset, MultiLevelDataset)
        self.assertEqual(len(level_chunks), ml_dataset.num_levels)
        for level in range(ml_dataset.num_levels):
            level_dataset = ml_dataset.get_dataset(level)
            self.assertEqual(
                {"precipitation", "soil_moisture", "temperature"},
                set(level_dataset.data_vars.keys()),
            )
            for var_name, var in level_dataset.data_vars.items():
                var_chunks = level_chunks[level]
                self.assertEqual(var_chunks, var.chunks, f"{var_name} at level {level}")

    def _assert_result_not_ok(self, result, message_regex: str):
        self.assertEqual(1, result.exit_code)
        self.assertRegex(result.stderr, message_regex)

    def test_level_with_nc(self):
        result = self.invoke_cli(
            [
                "level",
                "-t",
                "45",
                "-o",
                LevelDataTest.TEST_OUTPUT,
                TEST_NC_FILE,
            ]
        )
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual(
            {".zlevels", "0.zarr", "1.zarr", "2.zarr"},
            set(os.listdir(LevelDataTest.TEST_OUTPUT)),
        )

    def test_level_with_zarr(self):
        result = self.invoke_cli(
            [
                "level",
                "-t",
                "45",
                "-o",
                LevelDataTest.TEST_OUTPUT,
                TEST_ZARR_DIR,
            ]
        )
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual(
            {".zlevels", "0.zarr", "1.zarr", "2.zarr"},
            set(os.listdir(LevelDataTest.TEST_OUTPUT)),
        )

    def test_level_with_zarr_agg_method(self):
        result = self.invoke_cli(
            [
                "level",
                "-t",
                "45",
                "-A",
                "auto",
                "-o",
                LevelDataTest.TEST_OUTPUT,
                TEST_ZARR_DIR,
            ]
        )
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual(
            {".zlevels", "0.zarr", "1.zarr", "2.zarr"},
            set(os.listdir(LevelDataTest.TEST_OUTPUT)),
        )

    def test_level_with_zarr_agg_methods(self):
        result = self.invoke_cli(
            [
                "level",
                "-t",
                "45",
                "-A",
                "precipitation=mean,temperature=max,soil_*=median",
                "-o",
                LevelDataTest.TEST_OUTPUT,
                TEST_ZARR_DIR,
            ]
        )
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual(
            {".zlevels", "0.zarr", "1.zarr", "2.zarr"},
            set(os.listdir(LevelDataTest.TEST_OUTPUT)),
        )

    def test_level_with_zarr_link(self):
        result = self.invoke_cli(
            [
                "level",
                "--link",
                "-t",
                "45",
                "-o",
                LevelDataTest.TEST_OUTPUT,
                TEST_ZARR_DIR,
            ]
        )
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual(
            {".zlevels", "0.link", "1.zarr", "2.zarr"},
            set(os.listdir(LevelDataTest.TEST_OUTPUT)),
        )

    def test_level_with_zarr_num_levels_max(self):
        result = self.invoke_cli(
            [
                "level",
                "-t",
                "45",
                "-n",
                "2",
                "-o",
                LevelDataTest.TEST_OUTPUT,
                TEST_ZARR_DIR,
            ]
        )
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(LevelDataTest.TEST_OUTPUT))
        self.assertEqual(
            {".zlevels", "0.zarr", "1.zarr"}, set(os.listdir(LevelDataTest.TEST_OUTPUT))
        )

    def test_invalid_inputs(self):
        result = self.invoke_cli(
            [
                "level",
                "-t",
                "a45",
                "-o",
                LevelDataTest.TEST_OUTPUT,
                TEST_NC_FILE,
            ]
        )
        self._assert_result_not_ok(
            result,
            "Error\\: Invalid tile sizes in TILE_SIZE found: "
            "invalid literal for int\\(\\) with base 10\\: 'a45'\n",
        )

        result = self.invoke_cli(
            [
                "level",
                "-t",
                "-3",
                "-o",
                LevelDataTest.TEST_OUTPUT,
                TEST_NC_FILE,
            ]
        )
        self._assert_result_not_ok(
            result,
            "Error\\: Invalid tile sizes in TILE_SIZE found\\: "
            "all items must be positive integer numbers\n",
        )

        result = self.invoke_cli(
            ["level", "-t", "45,45,45", "-o", LevelDataTest.TEST_OUTPUT, TEST_NC_FILE]
        )
        self._assert_result_not_ok(
            result, "Error\\: TILE_SIZE must have 2 tile sizes separated by ','\n"
        )

        result = self.invoke_cli(
            ["level", "-n", "0", "-o", LevelDataTest.TEST_OUTPUT, TEST_NC_FILE]
        )
        self._assert_result_not_ok(
            result, "NUM_LEVELS_MAX must be a positive integer\n"
        )

    def test_with_existing_output(self):
        # Product output
        self.invoke_cli(["level", TEST_ZARR_DIR, "--output", "my.levels"])
        # Product output once mor
        result = self.invoke_cli(["level", TEST_ZARR_DIR, "--output", "my.levels"])
        self._assert_result_not_ok(
            result, "Error: output 'my\\.levels' already exists\n"
        )
