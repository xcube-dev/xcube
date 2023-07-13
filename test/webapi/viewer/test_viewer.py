# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import unittest
from typing import Optional, Mapping, Any

import pytest

from xcube.core.new import new_cube
from xcube.server.api import ApiError
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.viewer import Viewer

STYLES_CONFIG = {
    "Styles": [
        {
            "Identifier": "SST",
            "ColorMappings": {
                "analysed_sst": {
                    "ValueRange": [270, 290],
                    "ColorBar": "inferno"
                }
            }
        }
    ]
}


class ViewerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.viewer: Optional[Viewer] = None

    def tearDown(self) -> None:
        if self.viewer is not None:
            self.viewer.stop_server()

    def get_viewer(self, server_config: Optional[Mapping[str, Any]] = None) \
            -> Viewer:
        self.viewer = Viewer(server_config=server_config)
        return self.viewer

    def test_start_and_stop_server(self):
        viewer = self.get_viewer()
        self.assertTrue(viewer.is_server_running)
        self.assertIsInstance(viewer.datasets_ctx, DatasetsContext)
        viewer.stop_server()
        self.assertFalse(viewer.is_server_running)

    def test_info(self):
        viewer = self.get_viewer()
        # Just a smoke test:
        viewer.info()  # will print something

    def test_show(self):
        viewer = self.get_viewer()
        # Just a smoke test:
        result = viewer.show()  # will show viewer
        if result is not None:
            from IPython.core.display import HTML
            self.assertIsInstance(result, HTML)

    def test_no_config(self):
        viewer = self.get_viewer()
        self.assertIsInstance(viewer.server_config, dict)
        self.assertIn("port", viewer.server_config)
        self.assertIn("address", viewer.server_config)
        self.assertIn("reverse_url_prefix", viewer.server_config)

    def test_with_config(self):
        viewer = self.get_viewer(STYLES_CONFIG)
        self.assertIsInstance(viewer.server_config, dict)
        self.assertIn("port", viewer.server_config)
        self.assertIn("address", viewer.server_config)
        self.assertIn("reverse_url_prefix", viewer.server_config)
        self.assertIn("Styles", viewer.server_config)
        self.assertEqual(STYLES_CONFIG["Styles"],
                         viewer.server_config["Styles"])

    def test_urls(self):
        viewer = self.get_viewer()

        self.assertIn("port", viewer.server_config)
        port = viewer.server_config["port"]
        reverse_url_prefix = viewer.server_config.get("reverse_url_prefix")

        if not reverse_url_prefix:
            expected_server_url = f"http://localhost:{port}"
            self.assertEqual(expected_server_url,
                             viewer.server_url)

            expected_viewer_url = f"{expected_server_url}/viewer/" \
                                  f"?serverUrl={expected_server_url}"
            self.assertEqual(expected_viewer_url,
                             viewer.viewer_url)
        else:
            self.assertIsInstance(viewer.server_url, str)
            self.assertIsInstance(viewer.viewer_url, str)
            self.assertIn(reverse_url_prefix, viewer.server_url)
            self.assertIn(viewer.server_url, viewer.viewer_url)

    def test_urls_with_jl_env_var(self):
        env_var_key = "XCUBE_JUPYTER_LAB_URL"
        env_var_value = os.environ.get(env_var_key)
        os.environ[env_var_key] = "http://xcube-test-lab/"

        try:
            viewer = self.get_viewer()
            self.assertTrue(viewer.server_url.startswith(
                "http://xcube-test-lab/proxy/"
            ))
            self.assertTrue(viewer.viewer_url.startswith(
                "http://xcube-test-lab/proxy/"
            ))
            self.assertTrue("/viewer/" in viewer.viewer_url)
        finally:
            if env_var_value is not None:
                os.environ[env_var_key] = env_var_value
            else:
                del os.environ[env_var_key]

    def test_add_and_remove_dataset(self):
        viewer = self.get_viewer()

        # Generate identifier and get title from dataset
        ds_id_1 = viewer.add_dataset(
            new_cube(variables={"analysed_sst": 280.},
                     title="My SST 1"),
        )
        self.assertIsInstance(ds_id_1, str)

        # Provide identifier and title
        ds_id_2 = viewer.add_dataset(
            new_cube(variables={"analysed_sst": 282.}),
            ds_id="my_sst_2",
            title="My SST 2"
        )
        self.assertEqual("my_sst_2", ds_id_2)

        ds_config_1 = self.viewer.datasets_ctx.get_dataset_config(ds_id_1)
        self.assertEqual({"Identifier": ds_id_1,
                          "Title": "My SST 1"},
                         ds_config_1)

        ds_config_2 = self.viewer.datasets_ctx.get_dataset_config(ds_id_2)
        self.assertEqual({"Identifier": ds_id_2,
                          "Title": "My SST 2"},
                         ds_config_2)

        self.viewer.remove_dataset(ds_id_1)
        with pytest.raises(ApiError.NotFound):
            self.viewer.datasets_ctx.get_dataset_config(ds_id_1)

        self.viewer.remove_dataset(ds_id_2)
        with pytest.raises(ApiError.NotFound):
            self.viewer.datasets_ctx.get_dataset_config(ds_id_2)

    def test_add_dataset_with_style(self):
        viewer = self.get_viewer(STYLES_CONFIG)

        ds_id = viewer.add_dataset(
            new_cube(variables={"analysed_sst": 280.}),
            title="My SST",
            style="SST"
        )

        ds_config = self.viewer.datasets_ctx.get_dataset_config(ds_id)
        self.assertEqual({"Identifier": ds_id,
                          "Title": "My SST",
                          "Style": "SST"},
                         ds_config)

    def test_add_dataset_with_color_mapping(self):
        viewer = self.get_viewer()

        ds_id = viewer.add_dataset(
            new_cube(variables={"analysed_sst": 280.}),
            title="My SST",
            color_mappings={
                "analysed_sst": {
                    "ValueRange": [280., 290.],
                    "ColorBar": "plasma"
                }
            },
        )

        ds_config = self.viewer.datasets_ctx.get_dataset_config(ds_id)
        self.assertEqual({"Identifier": ds_id,
                          "Title": "My SST",
                          "Style": ds_id},
                         ds_config)

