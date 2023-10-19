# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray

from test.webapi.ows.coverages.test_context import get_coverages_ctx
from xcube.server.api import ApiError
from xcube.webapi.ows.coverages.controllers import (
    get_coverage_as_json,
    get_coverage_data,
    get_crs_from_dataset,
    dtype_to_opengis_datatype,
    get_dataarray_description,
    get_units,
)


class CoveragesControllersTest(unittest.TestCase):
    def test_get_coverage_as_json(self):
        result = get_coverage_as_json(get_coverages_ctx().datasets_ctx, 'demo')
        self.assertIsInstance(result, dict)
        path = Path(__file__).parent / 'expected.json'
        # with open(path, mode="w") as fp:
        #    json.dump(result, fp, indent=2)
        with open(path, mode="r") as fp:
            expected_result = json.load(fp)
        self.assertEqual(expected_result, result)

    def test_get_coverage_data_tiff(self):
        query = dict(
            bbox=['52,1,51,2'],
            datetime=['2017-01-25'],
            properties=['conc_chl'],
        )
        result = get_coverage_data(
            get_coverages_ctx().datasets_ctx, 'demo', query, 'image/tiff'
        )
        with BytesIO(result) as fh:
            da = rioxarray.open_rasterio(fh)
            self.assertIsInstance(da, xr.DataArray)
            self.assertEqual(('band', 'y', 'x'), da.dims)
            self.assertEqual('Chlorophyll concentration', da.long_name)
            self.assertEqual((1, 400, 400), da.shape)

    def test_get_coverage_data_netcdf(self):
        query = dict(
            bbox=['52,1,51,2'],
            datetime=['2017-01-24/2017-01-27'],
            properties=['conc_chl,kd489'],
            crs=['EPSG:4326'],
        )
        result = get_coverage_data(
            get_coverages_ctx().datasets_ctx,
            'demo',
            query,
            'application/netcdf',
        )
        # We can't read this directly from memory: the netcdf4 engine only
        # reads from filesystem paths, the h5netcdf engine (which can read
        # from memory) isn't an xcube dependency, and the scipy engine only
        # handles NetCDF 3.

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, 'out.nc')
            with open(path, 'wb') as fh:
                fh.write(result)
            ds = xr.open_dataset(path)
            self.assertEqual(
                {'lat': 400, 'lon': 400, 'time': 2, 'bnds': 2}, ds.dims
            )
            self.assertEqual(['conc_chl', 'kd489'], list(ds.data_vars))
            self.assertEqual(
                [
                    'lat',
                    'lat_bnds',
                    'lon',
                    'lon_bnds',
                    'time',
                    'time_bnds',
                    'conc_chl',
                    'kd489',
                ],
                list(ds.variables),
            )
            ds.close()

    def test_get_coverage_data_png(self):
        query = dict(
            subset=['lat(51:52),lon(1:2),time(2017-01-25)'],
            properties=['conc_chl'],
        )
        result = get_coverage_data(
            get_coverages_ctx().datasets_ctx, 'demo', query, 'png'
        )
        with BytesIO(result) as fh:
            da = rioxarray.open_rasterio(fh, driver='PNG')
            self.assertIsInstance(da, xr.DataArray)
            self.assertEqual(('band', 'y', 'x'), da.dims)
            self.assertEqual((1, 400, 400), da.shape)

    def test_get_coverage_unsupported_type(self):
        with self.assertRaises(ApiError.UnsupportedMediaType):
            get_coverage_data(
                get_coverages_ctx().datasets_ctx, 'demo', {}, 'nonexistent'
            )

    def test_get_crs_from_dataset(self):
        ds = xr.Dataset({'crs': ([], None, {'spatial_ref': '3035'})})
        self.assertEqual('EPSG:3035', get_crs_from_dataset(ds))

    def test_dtype_to_opengis_datatype(self):
        expected = [
            (
                np.uint16,
                'http://www.opengis.net/def/dataType/OGC/0/unsignedShort',
            ),
            (np.int32, 'http://www.opengis.net/def/dataType/OGC/0/signedInt'),
            (np.datetime64, 'http://www.opengis.net/def/bipm/UTC'),
            (np.object_, ''),
        ]
        for dtype, opengis in expected:
            self.assertEqual(opengis, dtype_to_opengis_datatype(dtype))

    def test_get_dataarray_description(self):
        name = 'foo'
        da = xr.DataArray(data=[], coords=[('x', [])], dims=['x'], name=name)
        self.assertEqual(name, get_dataarray_description(da))

    def test_get_units(self):
        self.assertEqual(
            'unknown',
            get_units(xr.Dataset({'time': [1, 2, 3]}), 'time')
        )
