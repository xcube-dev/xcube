import os
import subprocess
import sys
import time
import unittest
import urllib
import urllib.error
import urllib.request

import moto.server

MOTO_SERVER_ENDPOINT_URL = f'http://127.0.0.1:5000'

MOTOSERVER_PATH = moto.server.__file__
MOTOSERVER_ARGS = [sys.executable, MOTOSERVER_PATH, 's3']


class S3Test(unittest.TestCase):
    _moto_server = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        """Mocked AWS Credentials for moto."""
        os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
        os.environ['AWS_SECURITY_TOKEN'] = 'testing'
        os.environ['AWS_SESSION_TOKEN'] = 'testing'

        cls._moto_server = subprocess.Popen(MOTOSERVER_ARGS)
        t0 = time.perf_counter()
        running = False
        while not running and time.perf_counter() - t0 < 60.0:
            try:
                with urllib.request.urlopen(MOTO_SERVER_ENDPOINT_URL, timeout=1.0):
                    running = True
                    print(f'moto_server started after {round(1000 * (time.perf_counter() - t0))} ms')

            except urllib.error.URLError as e:
                pass
        if not running:
            raise Exception(f'Failed to start moto server after {round(1000 * (time.perf_counter() - t0))} ms')

    def setUp(self) -> None:
        # see https://github.com/spulec/moto/issues/2288
        urllib.request.urlopen(urllib.request.Request(MOTO_SERVER_ENDPOINT_URL + '/moto-api/reset', method='POST'))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._moto_server.kill()
        super().tearDownClass()

