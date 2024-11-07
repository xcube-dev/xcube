# Copyright (c) 2018-2024 by xcube team and contributors
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


class ViewerExtRoutesTest(RoutesTestCase):
    def test_viewer_ext_root(self):
        response = self.fetch("/viewer/ext")
        self.assertResponseOK(response)

    def test_viewer_ext_contributions(self):
        response = self.fetch("/viewer/ext/contributions")
        self.assertResponseOK(response)

    def test_viewer_ext_layout(self):
        response = self.fetch("/viewer/ext/layout")
        self.assertResponseOK(response)

    def test_viewer_ext_callback(self):
        response = self.fetch("/viewer/ext/layout")
        self.assertResponseOK(response)
