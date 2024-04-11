# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


import unittest

import jsonschema.exceptions
import pytest

from xcube.webapi.viewer.config import CONFIG_SCHEMA


class ViewerConfigTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_validate_instance_ok(self):
        CONFIG_SCHEMA.validate_instance({})
        CONFIG_SCHEMA.validate_instance({"Viewer": {}})
        CONFIG_SCHEMA.validate_instance(
            {
                "Viewer": {
                    "Configuration": {},
                }
            }
        )
        CONFIG_SCHEMA.validate_instance(
            {
                "Viewer": {
                    "Configuration": {"Path": "s3://xcube-viewer-app/bc/dev/viewer/"},
                }
            }
        )

    # noinspection PyMethodMayBeStatic
    def test_validate_instance_fails(self):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance(
                {
                    "Viewer": {
                        "Config": {},
                    }
                }
            )

        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance(
                {
                    "Viewer": {
                        "Configuration": {
                            "Path": "s3://xcube-viewer-app/bc/dev/viewer/",
                            "Title": "Test!",
                        },
                    }
                }
            )
