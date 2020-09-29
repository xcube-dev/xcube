import os
import subprocess
import sys
import time
import unittest
import urllib
import urllib.error
import urllib.request

import moto.server

MOTOSERVER_ENDPOINT_URL = f'http://127.0.0.1:5000'

MOTOSERVER_PATH = moto.server.__file__
MOTOSERVER_ARGS = [sys.executable, MOTOSERVER_PATH, 's3']


class S3Test(unittest.TestCase):
    _moto_server = None

    @classmethod
    def setUpClass(cls) -> None:

        """Mocked AWS Credentials for moto."""
        os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
        os.environ['AWS_SECURITY_TOKEN'] = 'testing'
        os.environ['AWS_SESSION_TOKEN'] = 'testing'

        cls._moto_server = subprocess.Popen(MOTOSERVER_ARGS)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 60.0:
            try:
                with urllib.request.urlopen(MOTOSERVER_ENDPOINT_URL, timeout=1.0):
                    print(f'moto_server started after {round(1000 * (time.perf_counter() - t0))} ms')
                    return
            except urllib.error.URLError as e:
                pass
        raise Exception(f'Failed to start moto server after {round(1000 * (time.perf_counter() - t0))} ms')

    @classmethod
    def tearDownClass(cls) -> None:
        cls._moto_server.kill()
