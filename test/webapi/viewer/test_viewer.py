# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os
import unittest
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Optional, Any, Union

import pytest
from xcube.core.mldataset import BaseMultiLevelDataset

from xcube.core.new import new_cube
from xcube.server.api import ApiError
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.viewer import Viewer

STYLES_CONFIG = {
    "Styles": [
        {
            "Identifier": "SST",
            "ColorMappings": {
                "analysed_sst": {"ValueRange": [270, 290], "ColorBar": "inferno"}
            },
        }
    ]
}


class ViewerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer: Optional[Viewer] = None

    def tearDown(self) -> None:
        if self.viewer is not None:
            self.viewer.stop_server()

    def get_viewer(
        self,
        server_config: Optional[Mapping[str, Any]] = None,
        roots: Optional[Union[str, Iterable[str]]] = None,
        max_depth: Optional[int] = None,
    ) -> Viewer:
        self.viewer = Viewer(
            server_config=server_config, roots=roots, max_depth=max_depth
        )
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
        self.assertIsInstance(viewer.server_config.get("port"), int)
        self.assertIsInstance(viewer.server_config.get("address"), str)
        self.assertIsInstance(viewer.server_config.get("reverse_url_prefix"), str)

    def test_with_config(self):
        viewer = self.get_viewer({"port": 8888, **STYLES_CONFIG})
        self.assertIsInstance(viewer.server_config, dict)
        # Get rid of "reverse_url_prefix" as it depends on env vars
        # noinspection PyUnresolvedReferences
        self.assertIsInstance(viewer.server_config.pop("reverse_url_prefix", None), str)
        self.assertEqual(
            {
                "address": "0.0.0.0",
                "port": 8888,
                **STYLES_CONFIG,
            },
            viewer.server_config,
        )

    def test_with_root(self):
        viewer = self.get_viewer({"port": 8081}, roots="data")
        self.assertIsInstance(viewer.server_config, dict)
        # Get rid of "reverse_url_prefix" as it depends on env vars
        # noinspection PyUnresolvedReferences
        self.assertIsInstance(viewer.server_config.pop("reverse_url_prefix", None), str)
        self.assertEqual(
            {
                "address": "0.0.0.0",
                "port": 8081,
                "DataStores": [
                    {
                        "Identifier": "_root_0",
                        "StoreId": "file",
                        "StoreParams": {"max_depth": 1, "root": "data"},
                    }
                ],
            },
            viewer.server_config,
        )

    def test_with_roots(self):
        viewer = self.get_viewer(
            {"port": 8080}, roots=["data", "s3://xcube"], max_depth=2
        )
        self.assertIsInstance(viewer.server_config, dict)
        # Get rid of "reverse_url_prefix" as it depends on env vars
        # noinspection PyUnresolvedReferences
        self.assertIsInstance(viewer.server_config.pop("reverse_url_prefix", None), str)
        self.assertEqual(
            {
                "address": "0.0.0.0",
                "port": 8080,
                "DataStores": [
                    {
                        "Identifier": "_root_0",
                        "StoreId": "file",
                        "StoreParams": {"max_depth": 2, "root": "data"},
                    },
                    {
                        "Identifier": "_root_1",
                        "StoreId": "s3",
                        "StoreParams": {"max_depth": 2, "root": "xcube"},
                    },
                ],
            },
            viewer.server_config,
        )

    def test_urls(self):
        viewer = self.get_viewer()

        self.assertIn("port", viewer.server_config)
        port = viewer.server_config["port"]
        reverse_url_prefix = viewer.server_config.get("reverse_url_prefix")

        if not reverse_url_prefix:
            expected_server_url = f"http://localhost:{port}"
            self.assertEqual(expected_server_url, viewer.server_url)

            expected_viewer_url = (
                f"{expected_server_url}/viewer/" f"?serverUrl={expected_server_url}"
            )
            self.assertEqual(expected_viewer_url, viewer.viewer_url)
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
            self.assertTrue(
                viewer.server_url.startswith("http://xcube-test-lab/proxy/")
            )
            self.assertTrue(
                viewer.viewer_url.startswith("http://xcube-test-lab/proxy/")
            )
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
            new_cube(variables={"analysed_sst": 280.0}, title="My SST 1"),
        )
        self.assertIsInstance(ds_id_1, str)

        # Provide identifier and title
        ds_id_2 = viewer.add_dataset(
            new_cube(variables={"analysed_sst": 282.0}),
            ds_id="my_sst_2",
            title="My SST 2",
        )
        self.assertEqual("my_sst_2", ds_id_2)

        ds_config_1 = self.viewer.datasets_ctx.get_dataset_config(ds_id_1)
        self.assertEqual({"Identifier": ds_id_1, "Title": "My SST 1"}, ds_config_1)

        ds_config_2 = self.viewer.datasets_ctx.get_dataset_config(ds_id_2)
        self.assertEqual({"Identifier": ds_id_2, "Title": "My SST 2"}, ds_config_2)

        self.viewer.remove_dataset(ds_id_1)
        with pytest.raises(ApiError.NotFound):
            self.viewer.datasets_ctx.get_dataset_config(ds_id_1)

        self.viewer.remove_dataset(ds_id_2)
        with pytest.raises(ApiError.NotFound):
            self.viewer.datasets_ctx.get_dataset_config(ds_id_2)

    # Verifies https://github.com/xcube-dev/xcube/issues/1007
    def test_add_dataset_with_slash_path(self):
        viewer = self.get_viewer(STYLES_CONFIG)

        ml_ds = BaseMultiLevelDataset(
            new_cube(variables={"analysed_sst": 280.0}),
            ds_id="mybucket/mysst.levels",
        )
        ds_id = viewer.add_dataset(ml_ds, title="My SST")

        self.assertEqual("mybucket-mysst.levels", ds_id)

        ds_config = self.viewer.datasets_ctx.get_dataset_config(ds_id)
        self.assertEqual({"Identifier": ds_id, "Title": "My SST"}, ds_config)

    def test_add_dataset_with_style(self):
        viewer = self.get_viewer(STYLES_CONFIG)

        ds_id = viewer.add_dataset(
            new_cube(variables={"analysed_sst": 280.0}), title="My SST", style="SST"
        )

        ds_config = self.viewer.datasets_ctx.get_dataset_config(ds_id)
        self.assertEqual(
            {"Identifier": ds_id, "Title": "My SST", "Style": "SST"}, ds_config
        )

    def test_add_dataset_with_color_mapping(self):
        viewer = self.get_viewer()

        ds_id = viewer.add_dataset(
            new_cube(variables={"analysed_sst": 280.0}),
            title="My SST",
            color_mappings={
                "analysed_sst": {"ValueRange": [280.0, 290.0], "ColorBar": "plasma"}
            },
        )

        ds_config = self.viewer.datasets_ctx.get_dataset_config(ds_id)
        self.assertEqual(
            {"Identifier": ds_id, "Title": "My SST", "Style": ds_id}, ds_config
        )
