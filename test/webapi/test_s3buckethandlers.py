import unittest
import urllib.request

from xcube.api import read_cube

SKIP_HELP = 'Require: xcube serve -p 9999 -c xcube/webapi/res/demo/cube.zarr'
SERVER_URL = 'http://localhost:9999'
ENDPOINT_URL = SERVER_URL + '/s3bucket'


def is_server_running() -> bool:
    # noinspection PyBroadException
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=2.0) as response:
            response.read()
    except Exception:
        return False
    return 200 <= response.status_code < 400


XCUBE_SERVER_IS_RUNNING = is_server_running()


class S3BucketHandlersTest(unittest.TestCase):

    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_read_cube_from_xube_server(self):
        ds = read_cube('s3bucket/local.zarr', format_name='zarr', endpoint_url=SERVER_URL)
        self.assertIsNotNone(ds)

    def test_read_cube_from_obs(self):
        ds = read_cube('dcs4cop-obs-01/OLCI-SNS-RAW-CUBE-2.zarr', format_name='zarr',
                       endpoint_url='https://obs.eu-de.otc.t-systems.com')
        self.assertIsNotNone(ds)
