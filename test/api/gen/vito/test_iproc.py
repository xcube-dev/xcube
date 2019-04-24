import unittest

import numpy as np
import pandas as pd
import xarray as xr

from test.sampledata import create_s2plus_dataset
from xcube.api.gen.vito.iproc import VitoS2PlusInputProcessor
from xcube.util.timecoord import get_time_in_days_since_1970


class VitoS2PlusInputProcessorTest(unittest.TestCase):

    def setUp(self):
        self.processor = VitoS2PlusInputProcessor()

    def test_props(self):
        self.assertEqual('vito-s2plus-l2', self.processor.name)
        self.assertEqual('VITO Sentinel-2 Plus Level 2 NetCDF inputs', self.processor.description)
        self.assertEqual('netcdf4', self.processor.input_reader)

    def test_get_time_range(self):
        ds = create_s2plus_dataset()
        t1, t2 = self.processor.get_time_range(ds)
        self.assertEqual(get_time_in_days_since_1970("2018-08-02T10:59:38.888000Z"), t1)
        self.assertEqual(get_time_in_days_since_1970("2018-08-02T10:59:38.888000Z"), t2)

    def test_pre_process(self):
        ds1 = create_s2plus_dataset()
        ds2 = self.processor.pre_process(ds1)
        self.assertIs(ds1, ds2)

    def test_post_process(self):
        ds1 = create_s2plus_dataset()
        ds2 = self.processor.post_process(ds1)
        self.assertIs(ds1, ds2)


def create_default_dataset(time_mode: str = "time_bnds"):
    w = 7200
    h = 3600
    res = 180. / h
    lon = np.linspace(-180 + 0.5 * res, 180 - 0.5 * res, w)
    lat = np.linspace(-90 + 0.5 * res, 90 - 0.5 * res, h)
    time = np.array([pd.to_datetime("20100301T120000Z")])
    time_bnds = np.array([[pd.to_datetime("20100301T000000Z"), pd.to_datetime("20100301T235959Z")]])

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

    return xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
