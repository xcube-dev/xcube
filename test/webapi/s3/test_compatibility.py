import unittest
import urllib.request

import numpy as np
import pytest
import xarray as xr

from xcube.core.dsio import open_dataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import new_data_store

SKIP_HELP = ('Skipped, because server is not running:'
             ' $ xcube serve2 -vvv -c examples/serve/demo/config.yml')
SERVER_URL = 'http://localhost:8080'

SERVER_ENDPOINT_URL = f'{SERVER_URL}/s3'


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
class S3CompatibilityTest(unittest.TestCase):

    def test_s3_datasets_bucket(self):
        store = new_data_store(
            "s3",
            root="datasets",
            storage_options=dict(
                anon=True,
                client_kwargs=dict(
                    endpoint_url=SERVER_ENDPOINT_URL
                )
            )
        )

        expected_data_ids = {'cog_local.zarr',
                             'geotiff_local.zarr',
                             'local.zarr',
                             'local_1w.zarr',
                             'local_ts.zarr',
                             'remote.zarr',
                             'remote_1w.zarr'}
        self.assertEqual(expected_data_ids, set(store.get_data_ids()))
        self.assertEqual(expected_data_ids, set(store.list_data_ids()))

        ds = store.open_data('local.zarr')
        self.assertCubeOk(ds)

    def test_s3_pyramids_bucket(self):
        store = new_data_store(
            "s3",
            root="pyramids",
            storage_options=dict(
                anon=True,
                client_kwargs=dict(
                    endpoint_url=SERVER_ENDPOINT_URL
                )
            )
        )

        expected_data_ids = {'cog_local.levels',
                             'geotiff_local.levels',
                             'local.levels',
                             'local_1w.levels',
                             'local_ts.levels',
                             'remote.levels',
                             'remote_1w.levels'}
        self.assertEqual(expected_data_ids, set(store.get_data_ids()))
        self.assertEqual(expected_data_ids, set(store.list_data_ids()))

        ds = store.open_data('local.levels')
        self.assertIsInstance(ds, MultiLevelDataset)
        self.assertCubeOk(ds.get_dataset(0))

    def test_open_dataset_from_xcube_server_rel_path(self):
        ds = open_dataset('datasets/local.zarr',
                          format_name='zarr',
                          s3_kwargs={
                              'anon': True
                          },
                          s3_client_kwargs=dict(
                              endpoint_url=SERVER_ENDPOINT_URL
                          ))
        self.assertCubeOk(ds)

    # noinspection PyMethodMayBeStatic
    def test_open_dataset_from_xcube_server_abs_path_raises(self):
        with pytest.raises(OSError,
                           match=r"\[Errno 5\] An error occurred \(\) when"
                                 r" calling the ListObjectsV2 operation:"
                                 r" "):
            open_dataset('http://localhost:8080/s3/datasets/local.zarr',
                         format_name='zarr',
                         s3_kwargs={
                             'anon': True
                         })

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
