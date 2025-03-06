# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from test.webapi.helpers import RoutesTestCase


class ViewerRoutesTest(RoutesTestCase):
    def test_viewer(self):
        response = self.fetch("/viewer")
        self.assertResponseOK(response)

        response = self.fetch("/viewer/")
        self.assertResponseOK(response)

        response = self.fetch("/viewer/index.html")
        self.assertResponseOK(response)

        response = self.fetch("/viewer/manifest.json")
        self.assertResponseOK(response)

        response = self.fetch("/viewer/images/logo.png")
        self.assertResponseOK(response)


class ViewerConfigRoutesTest(RoutesTestCase):
    def test_viewer_config(self):
        response = self.fetch("/viewer/config/config.json")
        self.assertResponseOK(response)


class ViewerStateRoutesNoConfigTest(RoutesTestCase):
    def test_get(self):
        response = self.fetch("/viewer/state")
        self.assertEqual(501, response.status)
        self.assertEqual("Persistence not supported", response.reason)

    def test_put(self):
        response = self.fetch("/viewer/state", method="PUT", body={"state": 123})
        self.assertEqual(501, response.status)
        self.assertEqual("Persistence not supported", response.reason)


class ViewerStateRoutesTest(RoutesTestCase):
    def get_config_filename(self) -> str:
        return "config-persistence.yml"

    def test_get_and_put_states(self):
        response = self.fetch("/viewer/state")
        self.assertResponseOK(response)
        result = response.json()
        self.assertEqual({"keys": []}, result)

        response = self.fetch("/viewer/state", method="PUT", body={"state": "Hallo"})
        self.assertResponseOK(response)
        result = response.json()
        self.assertIsInstance(result, dict)
        self.assertIn("key", result)
        key1 = result["key"]

        response = self.fetch("/viewer/state", method="PUT", body={"state": "Hello"})
        self.assertResponseOK(response)
        result = response.json()
        self.assertIsInstance(result, dict)
        self.assertIn("key", result)
        key2 = result["key"]

        response = self.fetch("/viewer/state")
        self.assertResponseOK(response)
        result = response.json()
        self.assertEqual({key1, key2}, set(result["keys"]))

        response = self.fetch(f"/viewer/state?key={key1}")
        self.assertResponseOK(response)
        result = response.json()
        self.assertEqual({"state": "Hallo"}, result)

        response = self.fetch(f"/viewer/state?key={key2}")
        self.assertResponseOK(response)
        result = response.json()
        self.assertEqual({"state": "Hello"}, result)


class ViewerExtRoutesTest(RoutesTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = None

    def get_config_filename(self) -> str:
        return "config-panels.yml"

    def test_viewer_ext_contributions(self):
        response = self.fetch("/viewer/ext/contributions")
        self.assertResponseOK(response)
        result = response.json()
        self.assertEqual({"result": expected_contributions_result}, result)

    def test_viewer_ext_layout(self):
        response = self.fetch(
            "/viewer/ext/layout/panels/0",
            method="POST",
            body={
                "inputValues": [
                    "",  # dataset_id
                ]
            },
        )
        self.assertResponseOK(response)
        result = response.json()
        self.assertEqual({"result": expected_layout_result}, result)

    def test_viewer_ext_callback(self):
        response = self.fetch(
            "/viewer/ext/callback",
            method="POST",
            body={
                "callbackRequests": [
                    {
                        "contribPoint": "panels",
                        "contribIndex": 0,
                        "callbackIndex": 0,
                        "inputValues": [
                            "",  # dataset_id
                            True,  # opaque
                            1,  # color
                            "",  # info_text
                        ],
                    }
                ]
            },
        )
        self.assertResponseOK(response)
        result = response.json()
        self.assertEqual({"result": expected_callback_result}, result)


class ViewerExtRoutesWithDefaultConfigTest(RoutesTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = None

    def test_viewer_ext_contributions(self):
        response = self.fetch("/viewer/ext/contributions")
        self.assertResourceNotFoundResponse(response)

    def test_viewer_ext_layout(self):
        response = self.fetch(
            "/viewer/ext/layout/panels/0",
            method="POST",
            body={
                "inputValues": [
                    "",  # dataset_id
                ]
            },
        )
        self.assertResourceNotFoundResponse(response)

    def test_viewer_ext_callback(self):
        response = self.fetch(
            "/viewer/ext/callback",
            method="POST",
            body={
                "callbackRequests": [
                    {
                        "contribPoint": "panels",
                        "contribIndex": 0,
                        "callbackIndex": 0,
                        "inputValues": [
                            "",  # dataset_id
                            True,  # opaque
                            1,  # color
                            "",  # info_text
                        ],
                    }
                ]
            },
        )
        self.assertResourceNotFoundResponse(response)


expected_callback_result = [
    {
        "contribIndex": 0,
        "contribPoint": "panels",
        "stateChanges": [
            {
                "id": "info_text",
                "property": "text",
                "value": "The dataset is , the color is green and "
                "it is opaque. The length of the last "
                "info text was 0. The number of "
                "datasets is 1.",
            }
        ],
    }
]

expected_layout_result = {
    "children": [
        {"id": "opaque", "label": "Opaque", "type": "Checkbox", "value": False},
        {
            "id": "color",
            "label": "Color",
            "options": [[0, "red"], [1, "green"], [2, "blue"], [3, "yellow"]],
            "style": {"flexGrow": 0, "minWidth": 80},
            "type": "Select",
            "value": 0,
        },
        {
            "children": [
                "The dataset is , the color is red and it "
                "is not opaque. The length of the last "
                "info text was 0. The number of "
                "datasets is 1."
            ],
            "id": "info_text",
            "type": "Typography",
        },
    ],
    "style": {
        "display": "flex",
        "flexDirection": "column",
        "gap": "6px",
        "height": "100%",
        "width": "100%",
    },
    "type": "Box",
}

expected_contributions_result = {
    "extensions": [{"name": "my_ext", "version": "0.0.0", "contributes": ["panels"]}],
    "contributions": {
        "panels": [
            {
                "name": "my_ext.my_panel_a",
                "initialState": {
                    "visible": False,
                    "title": "Panel A",
                    "icon": "equalizer",
                    "position": 2,
                },
                "extension": "my_ext",
                "layout": {
                    "function": {
                        "name": "render_panel",
                        "parameters": [
                            {
                                "name": "dataset_id",
                                "schema": {"type": "string"},
                                "default": "",
                            }
                        ],
                        "return": {"schema": {"type": "object", "class": "Component"}},
                    },
                    "inputs": [
                        {
                            "id": "@app",
                            "property": "selectedDatasetId",
                            "noTrigger": True,
                        }
                    ],
                },
                "callbacks": [
                    {
                        "function": {
                            "name": "update_info_text",
                            "parameters": [
                                {
                                    "name": "dataset_id",
                                    "schema": {"type": "string"},
                                    "default": "",
                                },
                                {
                                    "name": "opaque",
                                    "schema": {"type": "boolean"},
                                    "default": False,
                                },
                                {
                                    "name": "color",
                                    "schema": {"type": "integer"},
                                    "default": 0,
                                },
                                {
                                    "name": "info_text",
                                    "schema": {"type": "string"},
                                    "default": "",
                                },
                            ],
                            "return": {"schema": {"type": "string"}},
                        },
                        "inputs": [
                            {"id": "@app", "property": "selectedDatasetId"},
                            {"id": "opaque", "property": "value"},
                            {"id": "color", "property": "value"},
                            {
                                "id": "info_text",
                                "property": "text",
                                "noTrigger": True,
                            },
                        ],
                        "outputs": [{"id": "info_text", "property": "text"}],
                    }
                ],
            },
            {
                "name": "my_ext.my_panel_b",
                "initialState": {
                    "visible": False,
                    "title": "Panel B",
                    "icon": None,
                    "position": None,
                },
                "extension": "my_ext",
                "layout": {
                    "function": {
                        "name": "render_panel",
                        "parameters": [
                            {
                                "name": "dataset_id",
                                "schema": {"type": "string"},
                                "default": "",
                            }
                        ],
                        "return": {"schema": {"type": "object", "class": "Component"}},
                    },
                    "inputs": [
                        {
                            "id": "@app",
                            "property": "selectedDatasetId",
                            "noTrigger": True,
                        }
                    ],
                },
                "callbacks": [
                    {
                        "function": {
                            "name": "update_info_text",
                            "parameters": [
                                {
                                    "name": "dataset_id",
                                    "schema": {"type": "string"},
                                    "default": "",
                                },
                                {
                                    "name": "opaque",
                                    "schema": {"type": "boolean"},
                                    "default": False,
                                },
                                {
                                    "name": "color",
                                    "schema": {"type": "integer"},
                                    "default": 0,
                                },
                                {
                                    "name": "info_text",
                                    "schema": {"type": "string"},
                                    "default": "",
                                },
                            ],
                            "return": {"schema": {"type": "string"}},
                        },
                        "inputs": [
                            {"id": "@app", "property": "selectedDatasetId"},
                            {"id": "opaque", "property": "value"},
                            {"id": "color", "property": "value"},
                            {
                                "id": "info_text",
                                "property": "text",
                                "noTrigger": True,
                            },
                        ],
                        "outputs": [{"id": "info_text", "property": "text"}],
                    }
                ],
            },
        ]
    },
}
