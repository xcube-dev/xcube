import itertools
import json
import math
import unittest
from collections.abc import MutableMapping
from typing import Dict, KeysView, Iterator, Sequence, Any

import cftime
import numpy as np
import pytest
import xarray as xr

from test.sampledata import create_highroc_dataset
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.core.select import select_label_subset
from xcube.core.select import select_spatial_subset
from xcube.core.select import select_subset
from xcube.core.select import select_temporal_subset
from xcube.core.select import select_variables_subset


class SelectVariablesSubsetTest(unittest.TestCase):
    def test_select_variables_subset_all(self):
        ds1 = create_highroc_dataset()
        # noinspection PyTypeChecker
        ds2 = select_variables_subset(ds1, None)
        self.assertIs(ds2, ds1)
        ds2 = select_variables_subset(ds1, ds1.data_vars.keys())
        self.assertIs(ds2, ds1)

    def test_select_variables_subset_none(self):
        ds1 = create_highroc_dataset()
        ds2 = select_variables_subset(ds1, [])
        self.assertEqual(0, len(ds2.data_vars))
        ds2 = select_variables_subset(ds1, ['bibo'])
        self.assertEqual(0, len(ds2.data_vars))

    def test_select_variables_subset_some(self):
        ds1 = create_highroc_dataset()
        self.assertEqual(36, len(ds1.data_vars))
        ds2 = select_variables_subset(ds1, ['conc_chl', 'c2rcc_flags', 'rtoa_10'])
        self.assertEqual(3, len(ds2.data_vars))


class SelectSpatialSubsetTest(unittest.TestCase):
    def test_select_spatial_subset_all_ij_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, ij_bbox=(0, 0, 4, 3))
        self.assertIs(ds2, ds1)

    def test_select_spatial_subset_some_ij_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, ij_bbox=(1, 1, 4, 3))
        self.assertEqual((2, 3), ds2.conc_chl.shape)

    def test_select_spatial_subset_none_ij_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, ij_bbox=(5, 6, 7, 8))
        self.assertEqual(None, ds2)
        ds2 = select_spatial_subset(ds1, ij_bbox=(-6, -4, 2, 2))
        self.assertEqual(None, ds2)

    def test_select_spatial_subset_all_xy_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, xy_bbox=(7.9, 53.9, 12., 56.4))
        self.assertIs(ds2, ds1)

    def test_select_spatial_subset_some_xy_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, xy_bbox=(8., 55, 10., 56.))
        self.assertEqual((3, 3), ds2.conc_chl.shape)

    def test_select_spatial_subset_none_xy_bbox(self):
        ds1 = create_highroc_dataset()
        ds2 = select_spatial_subset(ds1, xy_bbox=(13., 57., 15., 60.))
        self.assertEqual(None, ds2)
        ds2 = select_spatial_subset(ds1, xy_bbox=(5.5, 55, 6.5, 56))
        self.assertEqual(None, ds2)

    def test_select_spatial_subset_invalid_params(self):
        ds1 = create_highroc_dataset()
        with self.assertRaises(ValueError) as cm:
            select_spatial_subset(ds1, ij_bbox=(5, 6, 7, 8), xy_bbox=(0., 0., 1., 2.))
        self.assertEqual("Only one of ij_bbox and xy_bbox can be given", f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            select_spatial_subset(ds1)
        self.assertEqual("One of ij_bbox and xy_bbox must be given", f'{cm.exception}')

    def test_select_spatial_subset_descending_y_param(self):
        ds1 = new_cube(inverse_y=True)
        ds2 = select_spatial_subset(ds1, xy_bbox=(40., 40., 42., 42.))
        self.assertEqual(((2,), (2,)), (ds2.lon.shape, ds2.lat.shape))

    def test_select_spatial_subset_with_gm(self):
        ds1 = new_cube(inverse_y=True)
        ds2 = select_spatial_subset(ds1, xy_bbox=(40., 40., 42., 42.),
                                    grid_mapping=GridMapping.from_dataset(ds1))
        self.assertEqual(((2,), (2,)), (ds2.lon.shape, ds2.lat.shape))


class SelectTemporalSubsetTest(unittest.TestCase):
    def test_invalid_dataset(self):
        ds1 = xr.Dataset(dict(CHL=xr.DataArray([[1, 2], [2, 3]], dims=('lat', 'lon'))))
        with self.assertRaises(ValueError) as cm:
            select_temporal_subset(ds1, time_range=('2010-01-02', '2010-01-04'))
        self.assertEqual('cannot compute temporal subset: variable "time" not found in dataset',
                         f'{cm.exception}')

    def test_no_subset_for_open_interval(self):
        ds1 = new_cube(variables=dict(analysed_sst=0.6, mask=8))
        ds2 = select_temporal_subset(ds1, time_range=(None, None))
        self.assertIs(ds2, ds1)

    def test_subset_for_closed_interval(self):
        ds1 = new_cube(variables=dict(analysed_sst=0.6, mask=8))
        ds2 = select_temporal_subset(ds1, time_range=('2010-01-02', '2010-01-04'))
        self.assertIsNot(ds2, ds1)
        np.testing.assert_equal(np.array(['2010-01-02T12:00:00.000000000',
                                          '2010-01-03T12:00:00.000000000',
                                          '2010-01-04T12:00:00.000000000'],
                                         dtype='datetime64[ns]'),
                                ds2.time.values)

    def test_subset_for_upward_open_interval(self):
        ds1 = new_cube(variables=dict(analysed_sst=0.6, mask=8))
        ds2 = select_temporal_subset(ds1, time_range=('2010-01-03', None))
        self.assertIsNot(ds2, ds1)
        np.testing.assert_equal(np.array(['2010-01-03T12:00:00.000000000',
                                          '2010-01-04T12:00:00.000000000',
                                          '2010-01-05T12:00:00.000000000'],
                                         dtype='datetime64[ns]'),
                                ds2.time.values)

    def test_subset_for_downward_open_interval(self):
        ds1 = new_cube(variables=dict(analysed_sst=0.6, mask=8))
        ds2 = select_temporal_subset(ds1, time_range=(None, '2010-01-03'))
        self.assertIsNot(ds2, ds1)
        np.testing.assert_equal(np.array(['2010-01-01T12:00:00.000000000',
                                          '2010-01-02T12:00:00.000000000',
                                          '2010-01-03T12:00:00.000000000'],
                                         dtype='datetime64[ns]'),
                                ds2.time.values)

    def test_cf_time_subset(self):
        ds1 = new_cube(variables=dict(analysed_sst=0.6, mask=8),
                       use_cftime=True,
                       time_dtype=None,
                       time_units='days since 1950-01-01',
                       time_calendar='julian')
        ds2 = select_temporal_subset(ds1, time_range=('2010-01-02', '2010-01-04'))
        self.assertIsNot(ds2, ds1)
        np.testing.assert_equal(np.array([cftime.DatetimeJulian(2010, 1, 2, 12, 0, 0),
                                          cftime.DatetimeJulian(2010, 1, 3, 12, 0, 0),
                                          cftime.DatetimeJulian(2010, 1, 4, 12, 0, 0)],
                                         dtype='object'),
                                ds2.time.values)


class SelectSubsetTest(unittest.TestCase):
    def test_all_params(self):
        ds1 = new_cube(variables=dict(analysed_sst=0.6, mask=8), drop_bounds=True)
        ds2 = select_subset(ds1,
                            var_names=['analysed_sst'],
                            bbox=(10, 50, 15, 55),
                            time_range=('2010-01-02', '2010-01-04'))

        self.assertEqual({'analysed_sst'}, set(ds2.data_vars))
        self.assertEqual({'time', 'lat', 'lon'}, set(ds2.coords))

        np.testing.assert_almost_equal(np.array([10.5, 11.5, 12.5, 13.5, 14.5]),
                                       ds2.lon.values)

        np.testing.assert_almost_equal(np.array([50.5, 51.5, 52.5, 53.5, 54.5]),
                                       ds2.lat.values)

        np.testing.assert_equal(np.array(['2010-01-02T12:00:00.000000000',
                                          '2010-01-03T12:00:00.000000000',
                                          '2010-01-04T12:00:00.000000000'],
                                         dtype='datetime64[ns]'),
                                ds2.time.values)

    def test_xy_bbox_with_large_dataset(self):
        ds = self._new_large_dataset()
        ds_subset = select_spatial_subset(ds, xy_bbox=(0., 0., 5.0, 2.5))
        self._assert_large_dataset_subset(ds_subset)

    def test_ij_bbox_with_large_dataset(self):
        ds = self._new_large_dataset()
        ds_subset = select_spatial_subset(ds, ij_bbox=(129600 // 2, 64800 // 2, 129600 // 2 + 1800, 64800 // 2 + 900))
        self._assert_large_dataset_subset(ds_subset)

    def _new_large_dataset(self):
        ds = new_virtual_dataset(lon_size=129600, lat_size=64800, time_size=5,
                                 lon_chunk=1296, lat_chunk=648, time_chunk=1)
        # print(ds)
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn('conc_chl', ds)
        self.assertEqual(('time', 'lat', 'lon'), ds.conc_chl.dims)
        self.assertEqual((5, 64800, 129600), ds.conc_chl.shape)
        self.assertEqual((5 * (1,), 100 * (648,), 100 * (1296,)),
                         ds.conc_chl.chunks)
        return ds

    def _assert_large_dataset_subset(self, ds_subset):
        # print(ds_subset)
        self.assertIsInstance(ds_subset, xr.Dataset)
        self.assertIn('conc_chl', ds_subset)
        self.assertEqual(('time', 'lat', 'lon'), ds_subset.conc_chl.dims)
        self.assertEqual((5, 900, 1800), ds_subset.conc_chl.shape)
        self.assertEqual(((1, 1, 1, 1, 1), (648, 252), (1296, 504)),
                         ds_subset.conc_chl.chunks)


class SelectLabelSubsetTest(unittest.TestCase):
    def test_predicate_callback(self):
        ds = xr.Dataset(
            {
                "a": (["time", "y", "x"], np.zeros((3, 4, 8))),
                "b": (["time", "y", "x"], np.ones((3, 4, 8))),
                "time": (["time"], [2020, 2021, 2022])
            }
        )

        actual_count = 0

        def predicate(slice_array, slice_info):
            nonlocal actual_count
            actual_count += 1
            self.assertIsInstance(slice_array, xr.DataArray)
            self.assertIsInstance(slice_info, dict)
            self.assertIn(slice_info.get('var'), ("a", "b"))
            self.assertIsInstance(slice_info.get('index'), int)
            self.assertIsInstance(slice_info.get('label'), xr.DataArray)
            self.assertEqual("time", slice_info.get('dim'))
            return True

        select_label_subset(ds, "time", predicate)
        self.assertEqual(3 * 2, actual_count)

    def test_no_change(self):
        self._test_no_change(use_dask=False)
        self._test_no_change(use_dask=True)

    def _test_no_change(self, use_dask: bool):
        ds = xr.Dataset(
            {
                "a": (["time", "y", "x"], np.zeros((3, 4, 8))),
                "b": (["time", "y", "x"], np.ones((3, 4, 8))),
                "time": (["time"], [2020, 2021, 2022])
            }
        )

        # noinspection PyUnusedLocal
        def predicate(slice_array, slice_info):
            return True

        result = select_label_subset(ds, "time", predicate, use_dask=use_dask)
        self.assertIs(ds, result)
        self.assertEqual({"a", "b", "time"}, set(result.variables.keys()))
        self.assertEqual({'time': 3, 'y': 4, 'x': 8}, result.dims)

    def test_select_by_slice_info(self):
        self._test_select_by_slice_info(use_dask=False)
        self._test_select_by_slice_info(use_dask=True)

    def _test_select_by_slice_info(self, use_dask: bool):
        ds = xr.Dataset(
            {
                "a": (["time", "y", "x"], np.zeros((3, 4, 8))),
                "b": (["time", "y", "x"], np.ones((3, 4, 8))),
                "time": (["time"], [2020, 2021, 2022])
            }
        )

        # noinspection PyUnusedLocal
        def predicate(slice_array, slice_info):
            return slice_info.get('label') == 2022

        result = select_label_subset(ds, "time", predicate, use_dask=use_dask)
        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual({'time': 1, 'y': 4, 'x': 8}, result.dims)
        self.assertEqual([2022], list(result.time))

        # noinspection PyUnusedLocal
        def predicate(slice_array, slice_info):
            return slice_info.get('label') in (2020, 2022)

        result = select_label_subset(ds, "time", predicate, use_dask=use_dask)
        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual({'time': 2, 'y': 4, 'x': 8}, result.dims)
        self.assertEqual([2020, 2022], list(result.time))

    def test_select_by_slice_array(self):
        self._test_select_by_slice_array(use_dask=False)
        self._test_select_by_slice_array(use_dask=True)

    def _test_select_by_slice_array(self, use_dask: bool):
        ds = xr.Dataset(
            {
                "a": (["time", "y", "x"], np.stack([np.full((4, 8), 1),
                                                    np.full((4, 8), 2),
                                                    np.full((4, 8), 3)])),
                "b": (["time", "y", "x"], np.stack([np.full((4, 8), 1),
                                                    np.full((4, 8), 2),
                                                    np.full((4, 8), 3)])),
                "time": (["time"], [2020, 2021, 2022])
            }
        )

        # noinspection PyUnusedLocal
        def predicate(slice_array, slice_info):
            return not np.all(slice_array == 2)

        result = select_label_subset(ds, "time", predicate, use_dask=use_dask)
        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual({'time': 2, 'y': 4, 'x': 8}, result.dims)
        self.assertEqual([2020, 2022], list(result.time))

        # noinspection PyUnusedLocal
        def predicate_a(slice_array, slice_info):
            return not np.all(slice_array == 2)

        # noinspection PyUnusedLocal
        def predicate_b(slice_array, slice_info):
            return not np.all(slice_array == 2)

        result = select_label_subset(ds, "time",
                                     dict(a=predicate_a, b=predicate_b),
                                     use_dask=use_dask)
        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual({'time': 2, 'y': 4, 'x': 8}, result.dims)
        self.assertEqual([2020, 2022], list(result.time))

    # noinspection PyMethodMayBeStatic
    def test_invalid_call(self):
        ds = xr.Dataset(
            {
                "a": (["time", "y", "x"], np.zeros((3, 4, 8))),
                "b": (["time", "y", "x"], np.ones((3, 4, 8))),
                "time": (["time"], [2020, 2021, 2022])
            }
        )

        with pytest.raises(TypeError,
                           match="predicate for variable 'a'"
                                 " must be callable with signature"):
            # noinspection PyTypeChecker
            select_label_subset(ds, "time", dict(a=137))

        with pytest.raises(TypeError,
                           match="predicate"
                                 " must be callable with signature"):
            # noinspection PyTypeChecker
            select_label_subset(ds, "time", 137)


def new_virtual_dataset(lon_size: int = 360,
                        lat_size: int = 180,
                        time_size: int = 5,
                        lon_chunk: int = None,
                        lat_chunk: int = None,
                        time_chunk: int = None):
    lon_res = 360. / lon_size
    lat_res = 180. / lat_size

    lon_values = np.linspace(-180. + lon_res / 2, 180. - lon_res / 2, lon_size)
    lat_values = np.linspace(-90. + lat_res / 2, 90. - lat_res / 2, lat_size)
    day1 = 365 * (2021 - 1970)
    time_values = np.linspace(day1, day1 + time_size, time_size, dtype='int64')

    chunk_store = VirtualChunkStore()
    chunk_store.add_unchunked_array('lon', ['lon'], lon_values)
    chunk_store.add_unchunked_array('lat', ['lat'], lat_values)
    chunk_store.add_unchunked_array('time', ['time'], time_values, attrs={
        "calendar": "gregorian",
        "units": "days since 1970-01-01"
    })
    chunk_store.add_chunked_array('conc_chl',
                                  ['time', 'lat', 'lon'],
                                  [time_size, lat_size, lon_size],
                                  [time_chunk or time_size, lat_chunk or lat_size, lon_chunk or lon_size])

    return xr.open_zarr(chunk_store)


class VirtualChunkStore(MutableMapping):

    def __init__(self, entries: Dict[str, bytes] = None):
        if entries:
            self._entries: Dict[str, bytes] = dict(entries)
        else:
            self._entries: Dict[str, bytes] = {
                ".zgroup": bytes(json.dumps({"zarr_format": 2}, indent=2), encoding='utf-8'),
                ".zattrs": bytes(json.dumps({}, indent=2), encoding='utf-8'),
            }

    def add_unchunked_array(self,
                            name: str,
                            dims: Sequence[str],
                            values: np.ndarray,
                            attrs: Dict[str, Any] = None):
        self._entries.update(self.get_array_entries_unchunked(name, dims, values, attrs=attrs))

    def add_chunked_array(self,
                          name: str,
                          dims: Sequence[str],
                          shape: Sequence[int],
                          chunks: Sequence[int],
                          value: float = None):
        self._entries.update(self.get_array_entries_chunked(name, dims, shape, chunks, value=value))

    def keys(self) -> KeysView[str]:
        return self._entries.keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> bytes:
        return self._entries[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        self._entries[key] = value

    def __delitem__(self, key: str) -> None:
        del self._entries[key]

    @classmethod
    def get_array_entries_unchunked(cls,
                                    name: str,
                                    dims: Sequence[str],
                                    values: np.ndarray,
                                    attrs: Dict[str, Any] = None) -> Dict[str, bytes]:

        zarray = {
            "zarr_format": 2,
            "chunks": list(values.shape),
            "shape": list(values.shape),
            "dtype": str(values.dtype.str),
            "order": "C",
            "fill_value": None,
            "compressor": None,
            "filters": None,
        }

        zattrs = {
            # For xarray
            "_ARRAY_DIMENSIONS": dims,
        }
        if attrs:
            zattrs.update(attrs)

        return {
            name + '/.zarray': bytes(json.dumps(zarray, indent=2), encoding='utf-8'),
            name + '/.zattrs': bytes(json.dumps(zattrs, indent=2), encoding='utf-8'),
            name + '/0': values.tobytes(order="C")
        }

    @classmethod
    def get_array_entries_chunked(cls,
                                  name: str,
                                  dims: Sequence[str],
                                  shape: Sequence[int],
                                  chunks: Sequence[int],
                                  value: float = None) -> Dict[str, bytes]:
        dtype = np.dtype('float64')

        zarray = {
            "zarr_format": 2,
            "dtype": str(dtype.str),
            "shape": list(shape),
            "chunks": list(chunks),
            "order": "C",
            "fill_value": "NaN",
            "compressor": None,
            "filters": None,
        }

        zattrs = {
            # For xarray
            "_ARRAY_DIMENSIONS": dims
        }

        entries = {
            name + '/.zarray': bytes(json.dumps(zarray, indent=2), encoding='utf-8'),
            name + '/.zattrs': bytes(json.dumps(zattrs, indent=2), encoding='utf-8'),
        }

        if value is not None:
            values = np.full(shape, value, dtype=dtype)
            data = values.tobytes(order="C")
            chunk_indices = [tuple(range(math.ceil(s / c))) for s, c in zip(shape, chunks)]
            for index in itertools.product(*chunk_indices):
                key = f'{name}/{"/".join(index)}'
                entries[key] = data

        return entries
