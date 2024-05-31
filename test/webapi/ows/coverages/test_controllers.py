# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
import numpy as np
import pyproj
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
    is_xy_order,
    transform_bbox,
    get_coverage_rangetype_for_dataset,
)


class CoveragesControllersTest(unittest.TestCase):
    def test_get_coverage_as_json(self):
        result = get_coverage_as_json(get_coverages_ctx().datasets_ctx, "demo")
        self.assertIsInstance(result, dict)
        path = Path(__file__).parent / "expected.json"
        # with open(path, mode="w") as fp:
        #    json.dump(result, fp, indent=2)
        with open(path) as fp:
            expected_result = json.load(fp)
        self.assertEqual(expected_result, result)

    def test_get_coverage_data_tiff(self):
        query = {
            "bbox": ["51,1,52,2"],
            "bbox-crs": ["[EPSG:4326]"],
            "datetime": ["2017-01-25T00:00:00Z"],
            "properties": ["conc_chl"],
        }
        content, content_bbox, content_crs = get_coverage_data(
            get_coverages_ctx().datasets_ctx, "demo", query, "image/tiff"
        )
        with BytesIO(content) as fh:
            da = rioxarray.open_rasterio(fh)
            self.assertIsInstance(da, xr.DataArray)
            self.assertEqual(("band", "y", "x"), da.dims)
            self.assertEqual("Chlorophyll concentration", da.long_name)
            self.assertEqual((1, 400, 400), da.shape)

    def test_get_coverage_data_geo_subset(self):
        query = {
            "subset": ["Lat(51:52),Lon(1:2)"],
            "subset-crs": ["[EPSG:4326]"],
            "datetime": ["2017-01-25T00:00:00Z"],
            "properties": ["conc_chl"],
            "crs": ["[OGC:CRS84]"],
        }
        content, content_bbox, content_crs = get_coverage_data(
            get_coverages_ctx().datasets_ctx, "demo", query, "image/tiff"
        )
        with BytesIO(content) as fh:
            da = rioxarray.open_rasterio(fh)
            self.assertIsInstance(da, xr.DataArray)
            self.assertEqual(("band", "y", "x"), da.dims)
            self.assertEqual("Chlorophyll concentration", da.long_name)
            self.assertEqual((1, 400, 400), da.shape)

    def test_get_coverage_data_netcdf(self):
        crs = "OGC:CRS84"
        # Unscaled size is 400, 400
        query = {
            "bbox": ["1,51,2,52"],
            "datetime": ["2017-01-24T00:00:00Z/2017-01-27T00:00:00Z"],
            "properties": ["conc_chl,kd489"],
            "scale-axes": ["lat(2),lon(2)"],
            "crs": [crs],
        }
        content, content_bbox, content_crs = get_coverage_data(
            get_coverages_ctx().datasets_ctx,
            "demo",
            query,
            "application/netcdf",
        )

        self.assertEqual(pyproj.CRS(crs), content_crs)
        self.assertEqual(4, len(content_bbox))
        for i in range(4):
            self.assertAlmostEqual([1.0, 51.0, 2.0, 52.0][i], content_bbox[i], places=2)

        # We can't read this directly from memory: the netcdf4 engine only
        # reads from filesystem paths, the h5netcdf engine (which can read
        # from memory) isn't an xcube dependency, and the scipy engine only
        # handles NetCDF 3.
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "out.nc")
            with open(path, "wb") as fh:
                fh.write(content)
            ds = xr.open_dataset(path)
            self.assertEqual({"lat": 200, "lon": 200, "time": 2, "bnds": 2}, ds.sizes)
            self.assertEqual({"conc_chl", "kd489", "crs"}, set(ds.data_vars))
            self.assertEqual(
                {
                    "lat",
                    "lat_bnds",
                    "lon",
                    "lon_bnds",
                    "time",
                    "time_bnds",
                    "conc_chl",
                    "kd489",
                    "crs",
                },
                set(ds.variables),
            )
            ds.close()

    def test_get_coverage_data_time_slice_subset(self):
        query = {
            "bbox": ["1,51,2,52"],
            "subset": ['time("2017-01-24T00:00:00Z":"2017-01-27T00:00:00Z")'],
            "properties": ["conc_chl"],
            "scale-factor": [4],
        }
        content, content_bbox, content_crs = get_coverage_data(
            get_coverages_ctx().datasets_ctx,
            "demo",
            query,
            "application/netcdf",
        )

        self.assertEqual(4, len(content_bbox))
        for i in range(4):
            self.assertAlmostEqual([51.0, 1.0, 52.0, 2.0][i], content_bbox[i], places=1)

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "out.nc")
            with open(path, "wb") as fh:
                fh.write(content)
            ds = xr.open_dataset(path)
            self.assertEqual({"lat": 100, "lon": 100, "time": 2, "bnds": 2}, ds.sizes)
            self.assertEqual({"conc_chl", "spatial_ref"}, set(ds.data_vars))
            self.assertEqual(
                {
                    "lat",
                    "lat_bnds",
                    "lon",
                    "lon_bnds",
                    "time",
                    "time_bnds",
                    "conc_chl",
                    "spatial_ref",
                },
                set(ds.variables),
            )
            ds.close()

    def test_get_coverage_data_png(self):
        query = {
            "subset": ["lat(52:51),lon(1:2),time(2017-01-25)"],
            "properties": ["conc_chl"],
            "scale-size": ["lat(100),lon(100)"],
        }
        content, content_bbox, content_crs = get_coverage_data(
            get_coverages_ctx().datasets_ctx, "demo", query, "png"
        )
        with BytesIO(content) as fh:
            da = rioxarray.open_rasterio(fh, driver="PNG")
            self.assertIsInstance(da, xr.DataArray)
            self.assertEqual(("band", "y", "x"), da.dims)
            self.assertEqual((1, 100, 100), da.shape)

    def test_get_coverage_no_data(self):
        with self.assertRaises(ApiError.NotFound):
            get_coverage_data(
                get_coverages_ctx().datasets_ctx,
                "demo",
                {"bbox": ["170,1,171,2"]},
                "application/netcdf",
            )

    def test_get_coverage_too_large(self):
        with self.assertRaises(ApiError.ContentTooLarge):
            get_coverage_data(
                get_coverages_ctx().datasets_ctx,
                "demo",
                {"scale-factor": ["0.01"]},
                "application/netcdf",
            )

    def test_get_coverage_unsupported_type(self):
        with self.assertRaises(ApiError.UnsupportedMediaType):
            get_coverage_data(
                get_coverages_ctx().datasets_ctx, "demo", {}, "nonexistent"
            )

    def test_get_coverage_unparseable_request(self):
        with self.assertRaises(ApiError.BadRequest):
            get_coverage_data(
                get_coverages_ctx().datasets_ctx,
                "demo",
                {"bbox": ["not a valid bbox specifier"]},
                "application/netcdf",
            )

    def test_get_coverage_nonexistent_property(self):
        with self.assertRaises(ApiError.BadRequest):
            get_coverage_data(
                get_coverages_ctx().datasets_ctx,
                "demo",
                {"properties": ["not_a_real_property"]},
                "application/netcdf",
            )

    def test_get_coverage_datetime_no_time(self):
        class CtxMock:
            def __init__(self, ds: xr.Dataset):
                self.ds = ds.rename_vars(dict(time="nottime"))

            def get_ml_dataset(self, _):
                return self

            def get_dataset(self, _):
                return self.ds

        # noinspection PyTypeChecker
        content, content_bbox, content_crs = get_coverage_data(
            CtxMock(
                get_coverages_ctx().datasets_ctx.get_ml_dataset("demo").get_dataset(0)
            ),
            "demo",
            {
                "datetime": ["2017-01-16T10:09:21.834255872Z"],
                "bbox": ["1,51,2,52"],
            },
            "application/netcdf",
        )

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "out.nc")
            with open(path, "wb") as fh:
                fh.write(content)
            with xr.open_dataset(path) as ds:
                self.assertEqual(
                    {"lat": 400, "lon": 400, "time": 5, "bnds": 2}, ds.sizes
                )

    def test_get_crs_from_dataset(self):
        ds = xr.Dataset({"crs": ([], None, {"spatial_ref": "3035"})})
        self.assertEqual("EPSG:3035", get_crs_from_dataset(ds).to_string())

    def test_dtype_to_opengis_datatype(self):
        expected = [
            (
                np.uint16,
                "http://www.opengis.net/def/dataType/OGC/0/unsignedShort",
            ),
            (np.int32, "http://www.opengis.net/def/dataType/OGC/0/signedInt"),
            (np.datetime64, "http://www.opengis.net/def/bipm/UTC"),
            (np.object_, ""),
        ]
        for dtype, opengis in expected:
            self.assertEqual(opengis, dtype_to_opengis_datatype(dtype))

    def test_get_dataarray_description(self):
        name = "foo"
        da = xr.DataArray(data=[], coords=[("x", [])], dims=["x"], name=name)
        self.assertEqual(name, get_dataarray_description(da))

    def test_get_units(self):
        self.assertEqual("unknown", get_units(xr.Dataset({"time": [1, 2, 3]}), "time"))

    def test_is_xy(self):
        self.assertTrue(
            is_xy_order(
                pyproj.CRS(
                    """GEOGCRS["a_strange_crs",ENSEMBLE["foo",MEMBER["a"],MEMBER["b"],
            ELLIPSOID["WGS 84",6378137,298.3,LENGTHUNIT["metre",1]],
            ENSEMBLEACCURACY[2.0]],
            PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.017]],CS[ellipsoidal,2],
            AXIS["u (u)",south,ANGLEUNIT["degree",0.017]],
            AXIS["v (v)",south,ANGLEUNIT["degree",0.017]]]"""
                )
            )
        )

    def test_transform_bbox_same_crs(self):
        self.assertEqual(
            bbox := [1, 2, 3, 4],
            transform_bbox(bbox, crs := pyproj.CRS("EPSG:4326"), crs),
        )

    def test_get_coverage_rangetype_for_dataset(self):
        self.assertEqual(
            {
                "type": "DataRecord",
                "field": [
                    {
                        "description": "v",
                        "encodingInfo": {
                            "dataType": "http://www.opengis.net/def/"
                            "dataType/OGC/0/signedLong"
                        },
                        "name": "v",
                        "type": "Quantity",
                    }
                ],
            },
            get_coverage_rangetype_for_dataset(
                xr.Dataset(
                    {
                        "x": [1, 2, 3],
                        "v": (["x"], np.array([0, 0, 0], dtype=np.int64)),
                        "dimensionless1": ([], None),
                        "dimensionless2": ([], None),
                    }
                )
            ),
        )
