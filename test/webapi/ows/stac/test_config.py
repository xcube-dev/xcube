# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


import unittest

import jsonschema.exceptions
import pytest
from xcube.webapi.ows.stac.config import CONFIG_SCHEMA


class StacContextTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_validate_instance_ok(self):
        CONFIG_SCHEMA.validate_instance({})
        CONFIG_SCHEMA.validate_instance({"STAC": {}})
        CONFIG_SCHEMA.validate_instance(
            {
                "STAC": {
                    "Identifier": "my-catalogue",
                    "Title": "My Catalogue",
                    "Description": "My first tiny little Catalogue",
                }
            }
        )

    # noinspection PyMethodMayBeStatic
    def test_validate_instance_fails(self):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance(
                {
                    "STAC": {
                        "Id": "my-catalogue",
                        "Title": "My Catalogue",
                        "Description": "My first tiny little Catalogue",
                    }
                }
            )

        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance(
                {
                    "STAC": {
                        "Identifier": 12,
                        "Title": "My Catalogue",
                        "Description": "My first tiny little Catalogue",
                    }
                }
            )
