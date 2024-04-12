# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.testing import ServerTestCase


class ServerTestCaseTest(ServerTestCase):
    def test_demonstrate_usage(self):
        url = f"http://localhost:{self.port}/I_do_not_exist"
        response = self.http.request("GET", url)
        self.assertEqual(404, response.status)
