import subprocess
import sys
import time
import unittest
import urllib
import urllib.error
import urllib.request

import moto.server

ENDPOINT_PORT = 3000
ENDPOINT_URL = f'http://127.0.0.1:{ENDPOINT_PORT}'

MOTOSERVER_PATH = moto.server.__file__
MOTOSERVER_ARGS = [sys.executable, MOTOSERVER_PATH, 's3', f'-p{ENDPOINT_PORT}']


class S3Test(unittest.TestCase):
    _moto_server = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._moto_server = subprocess.Popen(MOTOSERVER_ARGS)
        t0 = time.perf_counter()
        for i in range(60):
            try:
                with urllib.request.urlopen(ENDPOINT_URL, timeout=1):
                    print(f'moto_server started after {round(1000 * (time.perf_counter() - t0))} ms')
                    return
            except urllib.error.URLError:
                pass
        raise Exception(f'Failed to start moto server after {round(1000 * (time.perf_counter() - t0))} ms')

    @classmethod
    def tearDownClass(cls) -> None:
        cls._moto_server.kill()
