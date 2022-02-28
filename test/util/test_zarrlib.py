#  The MIT License (MIT)
#  Copyright (c) 2022 by the xcube development team and contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
import json

import fsspec

from test.s3test import S3Test, MOTO_SERVER_ENDPOINT_URL
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.core.store.fs.registry import new_fs_data_store
import numpy as np
import s3fs
import zarr.convenience
import pyproj

class ZarrLibTest(S3Test):

    def test_that_spatial_ref_can_be_added(self):
        original_dataset = new_cube(x_name='x',
                                    y_name='y',
                                    x_start=0,
                                    y_start=0,
                                    x_res=10,
                                    y_res=10,
                                    x_units='metres',
                                    y_units='metres',
                                    width=100,
                                    height=100,
                                    variables=dict(A=1.3, B=8.3))

        root = 'eurodatacube-test/xcube-eea'
        storage_options = dict(
            anon=False,
            client_kwargs=dict(
                endpoint_url=MOTO_SERVER_ENDPOINT_URL,
            )
        )

        fs = fsspec.filesystem('s3', **storage_options)
        fs.mkdirs(root, exist_ok=True)

        data_store = new_fs_data_store('s3',
                                       root=root,
                                       storage_options=storage_options)

        data_id = 'test.zarr'
        data_store.write_data(original_dataset, data_id=data_id)

        opened_dataset = data_store.open_data(data_id)
        self.assertEqual(set(opened_dataset.variables),
                         set(original_dataset.variables))

        with self.assertRaises(ValueError) as cm:
            GridMapping.from_dataset(opened_dataset)
        self.assertEqual(
            ('cannot find any grid mapping in dataset',),
            cm.exception.args
        )

        path = f"{root}/{data_id}"

        spatial_ref_store = fs.get_mapper(f"{path}/spatial_ref", create=True)
        group_store = fs.get_mapper(path, create=True)

        crs = pyproj.CRS.from_string("EPSG:3035")
        attrs = crs.to_cf()
        attrs['_ARRAY_DIMENSIONS'] = []
        spatial_ref = zarr.open(spatial_ref_store,
                                mode='w',
                                dtype=np.uint8,
                                shape=())
        spatial_ref.attrs.update(**attrs)
        self.assertTrue(fs.exists(f"{path}/spatial_ref"))
        self.assertTrue(fs.exists(f"{path}/spatial_ref/.zarray"))
        self.assertTrue(fs.exists(f"{path}/spatial_ref/.zattrs"))

        zarr.convenience.consolidate_metadata(group_store)

        print(fs.ls(path))
        print(fs.ls(f"{path}/spatial_ref"))
        with fs.open(f"{path}/.zmetadata", "r") as fp:
            zmetadata = fp.read()
            # zmetadata = json.load(fp)
            print(zmetadata)
        with fs.open(f"{path}/.zmetadata", "r") as fp:
            zmetadata = fp.read()
            # zmetadata = json.load(fp)
            print(zmetadata)

        opened_dataset = data_store.open_data(data_id)

        gm = GridMapping.from_dataset(opened_dataset)
        self.assertIn("LAEA Europe", gm.crs.srs)
