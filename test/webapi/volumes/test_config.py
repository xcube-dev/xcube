# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import jsonschema
import pytest

# noinspection PyProtectedMember
from xcube.webapi.volumes.config import CONFIG_SCHEMA


class VolumesAccessConfigTest(unittest.TestCase):
    def test_config_ok(self):
        self.assertIsNone(CONFIG_SCHEMA.validate_instance({}))

        self.assertIsNone(CONFIG_SCHEMA.validate_instance({"VolumesAccess": {}}))

        self.assertIsNone(
            CONFIG_SCHEMA.validate_instance(
                {"VolumesAccess": {"MaxVoxelCount": 100**3}}
            )
        )

    def test_config_fails(self):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance({"VolumesAccess": {"MaxVoxelCount": 1}})

        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance({"VolumesAccess": {"MaxVoxelCount": True}})

        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance(
                {"VolumesAccess": {"MaxPoxelCount": 100**3}}
            )
