# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import importlib
import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import call, patch


class VersionTest(unittest.TestCase):
    def test_falls_back_to_xcube_core_distribution(self):
        version_module = importlib.import_module("xcube.version")

        try:
            with patch(
                "importlib.metadata.version",
                side_effect=[PackageNotFoundError, "1.2.3"],
            ) as get_version:
                importlib.reload(version_module)

            self.assertEqual("1.2.3", version_module.version)
            self.assertEqual(
                [call("xcube"), call("xcube-core")],
                get_version.call_args_list,
            )
        finally:
            importlib.reload(version_module)
