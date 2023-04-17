import os
import unittest

from xcube.core.store import new_data_store
from xcube.core.store.fs.impl.dataset import DatasetKerchunkFsDataAccessor


class DatasetKerchunkFsDataAccessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.accessor = DatasetKerchunkFsDataAccessor()

    def test_get_format_id(self):
        self.assertEqual('kerchunk', self.accessor.get_format_id())

    def test_open_data(self):
        base_dir = os.path.dirname(__file__)
        open_params = {
            'protocol': 'file',
            'root': os.path.join(base_dir, 'data')
        }
        kerchunk_ds = self.accessor.open_data('data/c.json', **open_params)
        self.assertIsNotNone(kerchunk_ds)

    def test_open_from_store(self):
        store = new_data_store('file', root='/home/tonio-bc/EOData')
        data_ids = list(store.get_data_ids(include_attrs=['title',
                           'verification_flags',
                           'data_type']))
        desc = store.describe_data(
            'ESACCI-LST-L3S-LST-IRCDR_-0.01deg_1MONTHLY_DAY-fv2.00.json'
        )

    def test_open_kerchunk(self):
        protocol = 'file'
        storage_options = None

        import fsspec
        import xarray as xr

        fs = fsspec.filesystem(
            protocol,
            use_listings_cache=False,
            **(storage_options or {}))

        data_id = 'c.json'
        compression = None

        ref = fs.open(data_id, compression=compression)

        target_protocol = None

        kerchunk_store = fsspec.get_mapper(
            "reference://", fo=ref, target_protocol=target_protocol,
            remote_options={}, target_options={}
        )

        dataset = xr.open_zarr(kerchunk_store, consolidated=False)

        self.assertIsNotNone(dataset)
