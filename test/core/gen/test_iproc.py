import unittest

import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.gen.iproc import DefaultInputProcessor, ReprojectionInfo
from xcube.core.timecoord import to_time_in_days_since_1970


class ReprojectionInfoTest(unittest.TestCase):
    def test_success(self):
        # noinspection PyTypeChecker
        ri = ReprojectionInfo(xy_names=['x', 'y'])
        self.assertEqual(('x', 'y'), ri.xy_names)
        self.assertEqual(None, ri.xy_tp_names)
        self.assertEqual(None, ri.xy_crs)
        self.assertEqual(None, ri.xy_gcp_step)
        self.assertEqual(None, ri.xy_tp_gcp_step)

        # noinspection PyTypeChecker
        ri = ReprojectionInfo(xy_names=['x', 'y'], xy_gcp_step=[2, 1])
        self.assertEqual(('x', 'y'), ri.xy_names)
        self.assertEqual(None, ri.xy_tp_names)
        self.assertEqual(None, ri.xy_crs)
        self.assertEqual((2, 1), ri.xy_gcp_step)
        self.assertEqual(None, ri.xy_tp_gcp_step)

        # noinspection PyTypeChecker
        ri = ReprojectionInfo(xy_gcp_step=4, xy_tp_gcp_step=2)
        self.assertEqual(None, ri.xy_names)
        self.assertEqual(None, ri.xy_tp_names)
        self.assertEqual(None, ri.xy_crs)
        self.assertEqual((4, 4), ri.xy_gcp_step)
        self.assertEqual((2, 2), ri.xy_tp_gcp_step)

    def test_fail(self):
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            ReprojectionInfo(xy_names=['x', ''])
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            ReprojectionInfo(xy_names=[0, 'y'])
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            ReprojectionInfo(xy_gcp_step=[7, 'y'])
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            ReprojectionInfo(xy_gcp_step=[7, 0])


class DefaultInputProcessorTest(unittest.TestCase):

    def setUp(self):
        self.processor = DefaultInputProcessor()

    def test_props(self):
        self.assertEqual('default', self.processor.name)
        self.assertEqual('Single-scene NetCDF/CF inputs in xcube standard format', self.processor.description)
        self.assertEqual('netcdf4', self.processor.input_reader)

        processor = DefaultInputProcessor(input_reader="zarr")
        self.assertEqual('zarr', processor.input_reader)

    def test_reprojection_info(self):
        # noinspection PyNoneFunctionAssignment
        reprojection_info = self.processor.get_reprojection_info(create_default_dataset())
        self.assertIsNotNone(reprojection_info)

    def test_get_time_range(self):
        ds = create_default_dataset(time_mode="time")
        t1, t2 = self.processor.get_time_range(ds)
        self.assertEqual(to_time_in_days_since_1970("20100301T120000Z"), t1)
        self.assertEqual(to_time_in_days_since_1970("20100301T120000Z"), t2)
        ds = create_default_dataset(time_mode="time_bnds")
        t1, t2 = self.processor.get_time_range(ds)
        self.assertEqual(to_time_in_days_since_1970("20100301T000000Z"), t1)
        self.assertEqual(to_time_in_days_since_1970("20100301T235959Z"), t2)
        ds = create_default_dataset(time_mode="time_coverage")
        t1, t2 = self.processor.get_time_range(ds)
        self.assertEqual(to_time_in_days_since_1970("20100301T000000Z"), t1)
        self.assertEqual(to_time_in_days_since_1970("20100301T235959Z"), t2)
        ds = create_default_dataset(time_mode="start_stop_time")
        t1, t2 = self.processor.get_time_range(ds)
        self.assertEqual(to_time_in_days_since_1970("20100301T000000Z"), t1)
        self.assertEqual(to_time_in_days_since_1970("20100301T235959Z"), t2)
        ds = create_default_dataset(time_mode="start_stop_date")
        t1, t2 = self.processor.get_time_range(ds)
        self.assertEqual(to_time_in_days_since_1970("20100301T000000Z"), t1)
        self.assertEqual(to_time_in_days_since_1970("20100301T235959Z"), t2)
        ds = create_default_dataset(time_mode="no_time")
        with self.assertRaises(ValueError) as cm:
            self.processor.get_time_range(ds)
        self.assertEqual("invalid input: missing time coverage information in dataset", f"{cm.exception}")

    def test_pre_process(self):
        ds1 = create_default_dataset(time_mode="time")
        ds2 = self.processor.pre_process(ds1)
        self.assertIsNot(ds1, ds2)
        ds1 = create_default_dataset(time_mode="time_bnds")
        ds2 = self.processor.pre_process(ds1)
        self.assertIsNot(ds1, ds2)
        ds1 = create_default_dataset(time_mode="time_coverage")
        ds2 = self.processor.pre_process(ds1)
        self.assertIs(ds1, ds2)
        ds1 = create_default_dataset(time_mode="start_stop_time")
        ds2 = self.processor.pre_process(ds1)
        self.assertIs(ds1, ds2)

    def test_post_process(self):
        ds1 = create_default_dataset()
        ds2 = self.processor.post_process(ds1)
        self.assertIs(ds1, ds2)


def create_default_dataset(time_mode: str = "time_bnds"):
    w = 7200
    h = 3600
    res = 180. / h
    lon = np.linspace(-180 + 0.5 * res, 180 - 0.5 * res, w)
    lat = np.linspace(-90 + 0.5 * res, 90 - 0.5 * res, h)
    time = np.array([pd.to_datetime("20100301T120000")], dtype="datetime64[ns]")
    time_bnds = np.array([[pd.to_datetime("20100301T000000"), pd.to_datetime("20100301T235959")]],
                         dtype="datetime64[ns]")

    coords = dict(
        lon=(("lon",), lon, dict(long_name="longitude", units="degrees_east")),
        lat=(("lat",), lat, dict(long_name="latitude", units="degrees_north")),
    )

    if time_mode == "time":
        coords.update(dict(
            time=(("time",), time,
                  dict(long_name="time", units="nanoseconds since 1970-01-01"))
        ))
        var_dims = ("time", "lat", "lon")
        var_shape = (1, h, w)
    elif time_mode == "time_bnds":
        coords.update(dict(
            time=(
                ("time",), time,
                dict(long_name="time", units="nanoseconds since 1970-01-01")),
            time_bnds=(
                ("time", "bnds"), time_bnds,
                dict(long_name="time bounds", units="nanoseconds since 1970-01-01")),
        ))
        var_dims = ("time", "lat", "lon")
        var_shape = (1, h, w)
    else:
        var_dims = ("lat", "lon")
        var_shape = (h, w)

    analysed_sst = np.zeros(shape=var_shape, dtype=np.float32)
    analysis_error = np.zeros(shape=var_shape, dtype=np.float32)
    mask = np.zeros(shape=var_shape, dtype=np.int32)
    data_vars = dict(
        analysed_sst=(var_dims, analysed_sst),
        analysis_error=(var_dims, analysis_error),
        mask=(var_dims, mask),
    )

    attrs = dict([
        ('title', 'ESA SST CCI OSTIA L4 product'),
        ('institution', 'ESACCI'),
        ('publisher_name', 'ESACCI'),
        ('processing_level', 'L4'),
        ('Metadata_Conventions', 'Unidata Dataset Discovery v1.0'),
        ('Conventions', 'CF-1.5, Unidata Observation Dataset v1.0'),
        ('geospatial_lat_max', 90.0),
        ('geospatial_lat_min', -90.0),
        ('geospatial_lon_max', 180.0),
        ('geospatial_lon_min', -180.0),
    ])
    if time_mode == "time_coverage":
        attrs.update(dict([
            ('time_coverage_start', '20100301T000000Z'),
            ('time_coverage_end', '20100301T235959Z'),
            ('time_coverage_duration', 'P1D'),
            ('time_coverage_resolution', 'P1D'),
        ]))
    elif time_mode == "start_stop_time":
        attrs.update(dict([
            ('start_time', '20100301T000000Z'),
            ('stop_time', '20100301T235959Z'),
        ]))

    elif time_mode == "start_stop_date":
        attrs.update(dict([
            ('start_date', '01-MAR-2010 00:00:00.000000'),
            ('stop_date', '01-MAR-2010 23:59:59.000000'),
        ]))

    return xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
