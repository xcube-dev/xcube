import unittest
import urllib.request

import numpy as np
import xarray as xr

from xcube.core.dsio import open_dataset
from xcube.core.store import new_data_store

SKIP_HELP = ('Skipped, because server is not running:'
             ' $ xcube serve2 -vvv -c examples/serve/demo/config.yml')
SERVER_URL = 'http://localhost:8080'


def is_server_running() -> bool:
    # noinspection PyBroadException
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=2.0) as response:
            response.read()
    except Exception:
        return False
    return 200 <= response.code < 400


XCUBE_SERVER_IS_RUNNING = is_server_running()


@unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
class S3BucketHandlersTest(unittest.TestCase):

    def test_s3_data_store(self):
        store = new_data_store(
            "s3",
            root="s3",
            storage_options=dict(
                anon=True,
                client_kwargs=dict(
                    endpoint_url=SERVER_URL
                )
            )
        )

        # ds = store.open_data('local')
        # self.assertCubeOk(ds)

        data_ids = store.get_data_ids()
        self.assertEqual({"local",
                          "remote"},
                         set(data_ids))

    def test_open_dataset_from_xcube_server_rel_path(self):
        ds = open_dataset('s3/local',
                          format_name='zarr',
                          s3_kwargs={
                              'anon': True
                          },
                          s3_client_kwargs=dict(endpoint_url=SERVER_URL))
        self.assertCubeOk(ds)

    def test_open_dataset_from_xcube_server_abs_path(self):
        ds = open_dataset('http://localhost:8080/s3/local',
                          format_name='zarr',
                          s3_kwargs={
                              'anon': True
                          })
        self.assertCubeOk(ds)

    def assertCubeOk(self, ds):
        self.assertIsInstance(ds, xr.Dataset)
        self.assertEqual((5, 1000, 2000), ds.conc_chl.shape)
        self.assertEqual(('time', 'lat', 'lon'), ds.conc_chl.dims)
        conc_chl_values = ds.conc_chl.values
        self.assertEqual((5, 1000, 2000), conc_chl_values.shape)
        self.assertAlmostEqual(0.00005656,
                               float(np.nanmin(conc_chl_values)),
                               delta=1e-6)
        self.assertAlmostEqual(22.4421215,
                               float(np.nanmax(conc_chl_values)),
                               delta=1e-6)
