# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import asyncio
import json
import socket
import threading
import time
import unittest
from abc import ABC
from contextlib import closing
from typing import IO, Any, Iterable

import urllib3
from urllib3.exceptions import MaxRetryError

from xcube.server.server import Server
from xcube.server.webservers.tornado import TornadoFramework
from xcube.util.extension import ExtensionRegistry

Body = dict | str | bytes | IO[Any] | Iterable[bytes]
JsonData = None | bool | int | float | str | list | dict


# taken from https://stackoverflow.com/a/45690594
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class ServerTestCase(unittest.TestCase, ABC):
    """Base test class for unit tests of server framework

    This class is in the main xcube codebase rather than among the unit
    tests because xcube server is a framework module and API extension
    developers should be able to make use of the testing module.
    """

    def setUp(self) -> None:
        self.port = find_free_port()
        config = self.get_config()
        config.update({"port": self.port, "address": "localhost"})
        self.add_config(config)
        asyncio.set_event_loop(asyncio.new_event_loop())
        er = self.get_extension_registry()
        self.add_extension(er)
        self.server = Server(
            framework=TornadoFramework(), config=config, extension_registry=er
        )
        tornado = threading.Thread(target=self.server.start)
        tornado.daemon = True
        tornado.start()

        self.http = urllib3.PoolManager()

        # Taking care that server is fully started before tests make requests.
        # Fixing https://github.com/dcs4cop/xcube/issues/899
        while True:
            try:
                self.fetch("/")
                break
            except (MaxRetryError, TimeoutError):
                time.sleep(0.1)

    def tearDown(self) -> None:
        self.server.stop()

    # noinspection PyMethodMayBeStatic
    def get_extension_registry(self) -> ExtensionRegistry:
        return ExtensionRegistry()

    def add_extension(self, er: ExtensionRegistry) -> None:
        pass

    # noinspection PyMethodMayBeStatic
    def get_config(self) -> dict[str, Any]:
        return {}

    def add_config(self, config: dict) -> None:
        pass

    def fetch(
        self,
        path: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: Body | None = None,
    ) -> urllib3.response.HTTPResponse:
        """Fetches the response for given request.

        Args:
            path: Route request path
            method: HTTP request method, default to "GET"
            headers: HTTP request headers
            body: HTTP request body

        Returns:
            A response object
        """
        if path.startswith("/"):
            path = path[1:]
        url = f"http://localhost:{self.port}/{path}"
        if isinstance(body, dict):
            body = bytes(json.dumps(body), "utf-8")
        return self.http.request(method, url, headers=headers, body=body)

    def fetch_json(
        self,
        path: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: Body | None = None,
    ) -> tuple[JsonData, int]:
        """Fetches the response's JSON value for given request.

        Args:
            path: Route request path
            method: HTTP request method, default to "GET"
            headers: HTTP request headers
            body: HTTP request body

        Returns:
            A json object parsed from response data
        """
        response = self.fetch(path, method=method, headers=headers, body=body)
        return json.loads(response.data), response.status

    def assertResponseOK(self, response: urllib3.response.HTTPResponse):
        """Assert that response has status code 200.

        Args:
            response: The response from a ``self.fetch()`` call
        """
        self.assertResponse(response, expected_status=200)

    def assertResourceNotFoundResponse(
        self, response: urllib3.response.HTTPResponse, expected_message: str = None
    ):
        """Assert that response
        has the given status code 404, and
        has the given error message *expected_message*, if given.

        Args:
            response: The response from a ``self.fetch()`` call
            expected_message: The expected error message or part of it.
        """
        self.assertResponse(
            response, expected_status=404, expected_message=expected_message
        )

    def assertBadRequestResponse(
        self, response: urllib3.response.HTTPResponse, expected_message: str = None
    ):
        """Assert that response
        has the given status code 400, and
        has the given error message *expected_message*, if given.

        Args:
            response: The response from a ``self.fetch()`` call
            expected_message: The expected error message or part of it.
        """
        self.assertResponse(
            response, expected_status=400, expected_message=expected_message
        )

    def assertResponse(
        self,
        response: urllib3.response.HTTPResponse,
        expected_status: int | None = None,
        expected_message: str | None = None,
    ):
        """Assert that response
        has the given status code *expected_status*, if given, and
        has the given error message *expected_message*, if given.

        Args:
            response: The response from a ``self.fetch()`` call
            expected_status: The expected status code.
            expected_message: The expected error message or part of it.
        """
        message = self.parse_error_message(response)
        if expected_status is not None:
            self.assertEqual(
                expected_status, response.status, msg=message or response.reason
            )
        if expected_message is not None:
            self.assertIsNotNone(message, msg="error message not available")
            self.assertIsInstance(
                message, str, msg=f"error message must be a string, was {message!r}"
            )
            self.assertIn(expected_message, message)

    @classmethod
    def parse_error_message(cls, response: urllib3.response.HTTPResponse) -> str | None:
        """Parse an error message from the given response object."""
        # noinspection PyBroadException
        try:
            result = json.loads(response.data)
            return result["error"]["message"]
        except BaseException:
            return None


# TODO (forman): Remove before final PR
ServerTest = ServerTestCase
