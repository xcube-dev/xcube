import os.path
from typing import List, Tuple

from test.cli.helpers import CliDataTest
from test.cli.helpers import CliTest
from test.cli.helpers import TEST_NC_FILE, TEST_ZARR_DIR


class TileTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['tile', '--help'])
        self.assertEqual(0, result.exit_code)


class LevelDataTest(CliDataTest):
    TEST_OUTPUT = "out.tiles"

    def outputs(self) -> List[str]:
        return [LevelDataTest.TEST_OUTPUT, 'my.tiles', 'config.yml']

    def time_periods(self) -> int:
        return 2

    def test_all_defaults(self):
        result = self.invoke_cli(['tile', TEST_ZARR_DIR])
        self._assert_result_ok(result, LevelDataTest.TEST_OUTPUT, [
            'out.tiles/precipitation/0/0/0/0.png',
            'out.tiles/precipitation/0/0/0/1.png',
            'out.tiles/precipitation/0/tilemapresource.xml',
            'out.tiles/precipitation/1/0/0/0.png',
            'out.tiles/precipitation/1/0/0/1.png',
            'out.tiles/precipitation/1/tilemapresource.xml',
            'out.tiles/precipitation/metadata.json',
            'out.tiles/soil_moisture/0/0/0/0.png',
            'out.tiles/soil_moisture/0/0/0/1.png',
            'out.tiles/soil_moisture/0/tilemapresource.xml',
            'out.tiles/soil_moisture/1/0/0/0.png',
            'out.tiles/soil_moisture/1/0/0/1.png',
            'out.tiles/soil_moisture/1/tilemapresource.xml',
            'out.tiles/soil_moisture/metadata.json',
            'out.tiles/temperature/0/0/0/0.png',
            'out.tiles/temperature/0/0/0/1.png',
            'out.tiles/temperature/0/tilemapresource.xml',
            'out.tiles/temperature/1/0/0/0.png',
            'out.tiles/temperature/1/0/0/1.png',
            'out.tiles/temperature/1/tilemapresource.xml',
            'out.tiles/temperature/metadata.json'
        ])

    def test_with_output(self):
        result = self.invoke_cli(['tile', TEST_ZARR_DIR, '--output', 'my.tiles', '-vvv'])
        self._assert_result_ok(result, 'my.tiles', [
            'my.tiles/precipitation/0/0/0/0.png',
            'my.tiles/precipitation/0/0/0/1.png',
            'my.tiles/precipitation/0/tilemapresource.xml',
            'my.tiles/precipitation/1/0/0/0.png',
            'my.tiles/precipitation/1/0/0/1.png',
            'my.tiles/precipitation/1/tilemapresource.xml',
            'my.tiles/precipitation/metadata.json',
            'my.tiles/soil_moisture/0/0/0/0.png',
            'my.tiles/soil_moisture/0/0/0/1.png',
            'my.tiles/soil_moisture/0/tilemapresource.xml',
            'my.tiles/soil_moisture/1/0/0/0.png',
            'my.tiles/soil_moisture/1/0/0/1.png',
            'my.tiles/soil_moisture/1/tilemapresource.xml',
            'my.tiles/soil_moisture/metadata.json',
            'my.tiles/temperature/0/0/0/0.png',
            'my.tiles/temperature/0/0/0/1.png',
            'my.tiles/temperature/0/tilemapresource.xml',
            'my.tiles/temperature/1/0/0/0.png',
            'my.tiles/temperature/1/0/0/1.png',
            'my.tiles/temperature/1/tilemapresource.xml',
            'my.tiles/temperature/metadata.json'
        ])

    def test_with_tile_size(self):
        result = self.invoke_cli(['tile', TEST_ZARR_DIR, '-t', '90'])
        self._assert_result_ok(result, LevelDataTest.TEST_OUTPUT, [
            'out.tiles/precipitation/0/0/0/0.png',
            'out.tiles/precipitation/0/0/1/0.png',
            'out.tiles/precipitation/0/1/0/0.png',
            'out.tiles/precipitation/0/1/0/1.png',
            'out.tiles/precipitation/0/1/1/0.png',
            'out.tiles/precipitation/0/1/1/1.png',
            'out.tiles/precipitation/0/1/2/0.png',
            'out.tiles/precipitation/0/1/2/1.png',
            'out.tiles/precipitation/0/1/3/0.png',
            'out.tiles/precipitation/0/1/3/1.png',
            'out.tiles/precipitation/0/tilemapresource.xml',
            'out.tiles/precipitation/1/0/0/0.png',
            'out.tiles/precipitation/1/0/1/0.png',
            'out.tiles/precipitation/1/1/0/0.png',
            'out.tiles/precipitation/1/1/0/1.png',
            'out.tiles/precipitation/1/1/1/0.png',
            'out.tiles/precipitation/1/1/1/1.png',
            'out.tiles/precipitation/1/1/2/0.png',
            'out.tiles/precipitation/1/1/2/1.png',
            'out.tiles/precipitation/1/1/3/0.png',
            'out.tiles/precipitation/1/1/3/1.png',
            'out.tiles/precipitation/1/tilemapresource.xml',
            'out.tiles/precipitation/metadata.json',
            'out.tiles/soil_moisture/0/0/0/0.png',
            'out.tiles/soil_moisture/0/0/1/0.png',
            'out.tiles/soil_moisture/0/1/0/0.png',
            'out.tiles/soil_moisture/0/1/0/1.png',
            'out.tiles/soil_moisture/0/1/1/0.png',
            'out.tiles/soil_moisture/0/1/1/1.png',
            'out.tiles/soil_moisture/0/1/2/0.png',
            'out.tiles/soil_moisture/0/1/2/1.png',
            'out.tiles/soil_moisture/0/1/3/0.png',
            'out.tiles/soil_moisture/0/1/3/1.png',
            'out.tiles/soil_moisture/0/tilemapresource.xml',
            'out.tiles/soil_moisture/1/0/0/0.png',
            'out.tiles/soil_moisture/1/0/1/0.png',
            'out.tiles/soil_moisture/1/1/0/0.png',
            'out.tiles/soil_moisture/1/1/0/1.png',
            'out.tiles/soil_moisture/1/1/1/0.png',
            'out.tiles/soil_moisture/1/1/1/1.png',
            'out.tiles/soil_moisture/1/1/2/0.png',
            'out.tiles/soil_moisture/1/1/2/1.png',
            'out.tiles/soil_moisture/1/1/3/0.png',
            'out.tiles/soil_moisture/1/1/3/1.png',
            'out.tiles/soil_moisture/1/tilemapresource.xml',
            'out.tiles/soil_moisture/metadata.json',
            'out.tiles/temperature/0/0/0/0.png',
            'out.tiles/temperature/0/0/1/0.png',
            'out.tiles/temperature/0/1/0/0.png',
            'out.tiles/temperature/0/1/0/1.png',
            'out.tiles/temperature/0/1/1/0.png',
            'out.tiles/temperature/0/1/1/1.png',
            'out.tiles/temperature/0/1/2/0.png',
            'out.tiles/temperature/0/1/2/1.png',
            'out.tiles/temperature/0/1/3/0.png',
            'out.tiles/temperature/0/1/3/1.png',
            'out.tiles/temperature/0/tilemapresource.xml',
            'out.tiles/temperature/1/0/0/0.png',
            'out.tiles/temperature/1/0/1/0.png',
            'out.tiles/temperature/1/1/0/0.png',
            'out.tiles/temperature/1/1/0/1.png',
            'out.tiles/temperature/1/1/1/0.png',
            'out.tiles/temperature/1/1/1/1.png',
            'out.tiles/temperature/1/1/2/0.png',
            'out.tiles/temperature/1/1/2/1.png',
            'out.tiles/temperature/1/1/3/0.png',
            'out.tiles/temperature/1/1/3/1.png',
            'out.tiles/temperature/1/tilemapresource.xml',
            'out.tiles/temperature/metadata.json'
        ])

    def _assert_result_ok(self, result, output_path: str, actual_file_listing: List[Tuple]):
        self.assertEqual(0, result.exit_code)
        # self.assertRegex(result.stdout, message_regex)
        self.assertTrue(os.path.isdir(output_path))

        expected_file_listing = []
        for root, dirs, files in os.walk(output_path, topdown=False):
            for name in files:
                expected_file_listing.append(os.path.join(root, name).replace('\\', '/'))
        expected_file_listing.sort()

        self.assertEqual(expected_file_listing, actual_file_listing)

    def test_invalid_inputs(self):
        result = self.invoke_cli(["tile",
                                  "-t", "a45",
                                  TEST_ZARR_DIR,
                                  ])
        self._assert_result_not_ok(result,
                                   "Error\\: Invalid tile sizes in TILE_SIZE found: "
                                   "invalid literal for int\\(\\) with base 10\\: 'a45'\n")

        result = self.invoke_cli(["tile",
                                  "-t", "-3",
                                  TEST_NC_FILE,
                                  ])
        self._assert_result_not_ok(result,
                                   "Error\\: Invalid tile sizes in TILE_SIZE found\\: "
                                   "all items must be positive integer numbers\n")

        result = self.invoke_cli(["tile",
                                  "-t", "45,45,45",
                                  TEST_NC_FILE])
        self._assert_result_not_ok(result,
                                   "Error\\: TILE_SIZE must have 2 tile sizes separated by ','\n")

    def _assert_result_not_ok(self, result, message_regex: str):
        self.assertEqual(1, result.exit_code)
        self.assertRegex(result.stderr, message_regex)
