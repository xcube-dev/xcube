import os
import subprocess
import sys
import time
import unittest
import urllib
import urllib.error
import urllib.request
from typing import Optional

import moto.server

MOTO_SERVER_ENDPOINT_URL = f'http://127.0.0.1:5000'

MOTO_SERVER_PATH = moto.server.__file__
MOTO_SERVER_ARGS = [sys.executable, MOTO_SERVER_PATH, 's3']

MOTO_SERVER_PROCESS: Optional[subprocess.Popen] = None


class S3Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        global MOTO_SERVER_PROCESS

        if MOTO_SERVER_PROCESS is not None:
            return

        """Mocked AWS Credentials for moto."""
        os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
        os.environ['AWS_SECURITY_TOKEN'] = 'testing'
        os.environ['AWS_SESSION_TOKEN'] = 'testing'

        t0 = time.perf_counter()
        print(f'running moto_server with args {MOTO_SERVER_ARGS}...')
        MOTO_SERVER_PROCESS = subprocess.Popen(MOTO_SERVER_ARGS)
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
