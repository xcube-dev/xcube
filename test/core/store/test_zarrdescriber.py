import unittest

import os
import s3fs
import shutil

from test.s3test import MOTO_SERVER_ENDPOINT_URL
from test.s3test import S3Test
from xcube.core.new import new_cube
from xcube.core.store import DatasetDescriptor
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store.zarrdescriber import DirectoryZarrDescriber
from xcube.core.store.zarrdescriber import S3ZarrDescriber


class S3ZarrDescriberTest(S3Test):

    BUCKET_NAME = 'zarr_describer_test'

    def setUp(self) -> None:
        super().setUp()
        dataset = new_cube(variables=dict(a=4.1, b=7.4))
        client_kwargs = dict(region_name='eu-central-1',
                             endpoint_url=MOTO_SERVER_ENDPOINT_URL)
        s3 = s3fs.S3FileSystem(key='test_fake_id',
                               secret='test_fake_secret',
                               client_kwargs=client_kwargs)
        s3.mkdir(self.BUCKET_NAME)
        dataset.to_zarr(s3fs.S3Map(root=f'{self.BUCKET_NAME}/cube',
                                   s3=s3,
                                   check=False))
        self._zarr_describer = S3ZarrDescriber(s3, self.BUCKET_NAME)

    def test_describe(self):
        descriptor = self._zarr_describer.describe('cube')
        self.assertIsInstance(descriptor, DatasetDescriptor)
        self.assertEqual('cube', descriptor.data_id)
        self.assertEqual(TYPE_SPECIFIER_DATASET, descriptor.type_specifier)
        self.assertEqual((-90, -180, 90, 180), descriptor.bbox)
        self.assertDictEqual(dict(lat=180, time=5, lon=360), descriptor.dims)
        self.assertEqual(('2010-01-01T00:00:00', '2010-01-06T00:00:00'), descriptor.time_range)


class DirectoryZarrDescriberTest(unittest.TestCase):

    PATH_NAME = 'zarr_describer_test'

    def setUp(self) -> None:
        super().setUp()
        dataset = new_cube(variables=dict(a=4.1, b=7.4))
        if os.path.exists(self.PATH_NAME):
            shutil.rmtree(self.PATH_NAME)
        os.mkdir(self.PATH_NAME)
        dataset.to_zarr(f'{self.PATH_NAME}/cube')
        self._zarr_describer = DirectoryZarrDescriber(self.PATH_NAME)

    def tearDown(self) -> None:
        shutil.rmtree(self.PATH_NAME)

    def test_describe(self):
        descriptor = self._zarr_describer.describe('cube')
        self.assertIsInstance(descriptor, DatasetDescriptor)
        self.assertEqual('cube', descriptor.data_id)
        self.assertEqual(TYPE_SPECIFIER_DATASET, descriptor.type_specifier)
        self.assertEqual((-90, -180, 90, 180), descriptor.bbox)
        self.assertDictEqual(dict(lat=180, time=5, lon=360), descriptor.dims)
        self.assertEqual(('2010-01-01T00:00:00', '2010-01-06T00:00:00'), descriptor.time_range)
