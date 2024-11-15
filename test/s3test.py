# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import contextlib
import functools
import os
import subprocess
import sys
import time
import unittest
import urllib
import urllib.error
import urllib.request
from typing import Callable

import moto.server

MOTO_SERVER_ENDPOINT_URL = f"http://127.0.0.1:5000"

MOTOSERVER_PATH = moto.server.__file__
MOTOSERVER_ARGS = [sys.executable, MOTOSERVER_PATH]

# Mocked AWS environment variables for Moto.
MOTOSERVER_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_ENDPOINT_URL_S3": MOTO_SERVER_ENDPOINT_URL,
}


class S3Test(unittest.TestCase):
    _stop_moto_server = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._stop_moto_server = _start_moto_server()

    def setUp(self) -> None:
        _reset_moto_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._stop_moto_server()
        super().tearDownClass()


def s3_test(timeout: float = 60.0, ping_timeout: float = 1.0):
    """A decorator to run individual tests with a Moto S3 server.

    The decorated tests receives the Moto Server's endpoint URL.

    Args:
        timeout:
            Total time in seconds it may take to start the server.
            Raises if time is exceeded.
        ping_timeout:
            Timeout for individual ping requests trying to access the
            started moto server.
            The total number of pings is ``int(timeout / ping_timeout)``.
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            stop_moto_server = _start_moto_server(timeout, ping_timeout)
            try:
                return test_func(*args, endpoint_url=MOTO_SERVER_ENDPOINT_URL, **kwargs)
            finally:
                stop_moto_server()

        return wrapper

    return decorator


@contextlib.contextmanager
def s3_test_server(timeout: float = 60.0, ping_timeout: float = 1.0) -> str:
    """A context manager that starts a Moto S3 server for testing.

    Args:
        timeout:
            Total time in seconds it may take to start the server.
            Raises if time is exceeded.
        ping_timeout:
            Timeout for individual ping requests trying to access the
            started moto server.
            The total number of pings is ``int(timeout / ping_timeout)``.

    Returns:
        The server's endpoint URL

    Raises:
        Exception: If the server could not be started or
            if the service is not available after after
            *timeout* seconds.
    """
    stop_moto_server = _start_moto_server(timeout=timeout, ping_timeout=ping_timeout)
    try:
        _reset_moto_server()
        yield MOTO_SERVER_ENDPOINT_URL
    finally:
        stop_moto_server()


def _start_moto_server(
    timeout: float = 60.0, ping_timeout: float = 1.0
) -> Callable[[], None]:
    """Start a Moto S3 server for testing.

    Args:
        timeout:
            Total time in seconds it may take to start the server.
            Raises if time is exceeded.
        ping_timeout:
            Timeout for individual ping requests trying to access the
            started moto server.
            The total number of pings is ``int(timeout / ping_timeout)``.

    Returns:
        A function that stops the server and restores the environment.

    Raises:
        Exception: If the server could not be started or
            if the service is not available after
            *timeout* seconds.
    """

    prev_env: dict[str, str | None] = {
        k: os.environ.get(k) for k, v in MOTOSERVER_ENV.items()
    }
    os.environ.update(MOTOSERVER_ENV)

    moto_server = subprocess.Popen(MOTOSERVER_ARGS)
    t0 = time.perf_counter()
    running = False
    while not running and time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(MOTO_SERVER_ENDPOINT_URL, timeout=ping_timeout):
                running = True
                print(
                    f"moto_server started after"
                    f" {round(1000 * (time.perf_counter() - t0))} ms"
                )

        except urllib.error.URLError:
            pass
    if not running:
        raise Exception(
            f"Failed to start moto server"
            f" after {round(1000 * (time.perf_counter() - t0))} ms"
        )

    def stop_moto_server():
        try:
            moto_server.kill()
        finally:
            # Restore environment variables
            for k, v in prev_env.items():
                if v is None:
                    del os.environ[k]
                else:
                    os.environ[k] = v

    return stop_moto_server


def _reset_moto_server():
    # see https://github.com/spulec/moto/issues/2288
    urllib.request.urlopen(
        urllib.request.Request(
            MOTO_SERVER_ENDPOINT_URL + "/moto-api/reset", method="POST"
        )
    )
